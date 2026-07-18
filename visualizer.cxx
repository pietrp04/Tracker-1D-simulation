#include "visualizer.h"
#include "config_new.h"
#include "detector.h"
#include "source.h"
#include "truth.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <TCanvas.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TLine.h>
#include <TBox.h>
#include <TMarker.h>
#include <TAxis.h>
#include <TF1.h>
#include <TStyle.h>
#include <TH1F.h>
#include <TPad.h>
#include <TLegend.h>

Visualizer::Visualizer(int argc, char* argv[]) {
    	m_app = new TApplication("app", &argc, argv);
}

Visualizer::~Visualizer() {
    	delete m_app;
}

TH1F* Visualizer::MakeRootHist(const QHistogram& h, const char* name, const char* title) {
    	TH1F* th = new TH1F(name, title, h.n_bins, h.q_min, h.q_max);
    	for (int b = 0; b < h.n_bins; b++) {
        	th->SetBinContent(b + 1, (double)h.counts[b]);
    	}
    	return th;
}

// Calibration plot for a single q histogram (used both for the clone-cleaned
// tracks and, with different arguments, for the raw uncleaned tracks).
// mu_q/sigma_q are estimated with RobustPeakEstimate (local FWHM + fixed-
// window recentering, no chi2/Minuit fit), which is robust when the physical
// sigma is much narrower than the search window (a least-squares fit can
// collapse on a degenerate minimum there). The red gaussian drawn on top is
// computed from the estimate, not fitted: it is only a visual reference, and
// is skipped entirely when draw_fit_curve=false (histogram shape alone).
// The estimate is always computed either way, so mu_q/sigma_q are always
// returned and printed to console.
// name_suffix keeps ROOT object names unique across repeated calls in the
// same run (e.g. "" for the cleaned histogram, "_raw" for the raw one,
// "_nofit" for the no-curve companion).
// Returns {mu_q, sigma_q}.
std::pair<float, float> Visualizer::DrawSourceSearchAndFit(const QHistogram& h, const char* name_suffix, const char* title, bool draw_fit_curve)
{
    	TH1F* hq = MakeRootHist(h, Form("hq_calib%s", name_suffix), title);

    	TCanvas* c0 = new TCanvas(Form("c0_calib%s", name_suffix), title, 100, 100, 900, 650);
    	c0->SetGrid();

    	hq->SetLineColor(kBlue);
    	hq->Draw("HIST");

    	// Theoretical geometric sigma: a-priori width that guides the adaptive
    	// bin aggregation inside RobustPeakEstimate when the histogram shows
    	// quantisation artefacts (isolated tall bins separated by exact zeros).
    	const float sigma_teorica = CalcolaTolleranzeTeoriche().sigma_q;
    	PeakEstimate stima = RobustPeakEstimate(h, sigma_teorica);

    	if (stima.valid && draw_fit_curve) {
    		TF1* curva = new TF1(Form("curva_calib%s", name_suffix), "gaus", h.q_min, h.q_max);
    		curva->SetParameters(hq->GetMaximum(), stima.mu, std::max(stima.sigma, h.bin_width()));
    		curva->SetLineColor(kRed);
    		curva->SetNpx(1000);
    		curva->Draw("SAME");
    	} else if (!stima.valid) {
    		std::cerr << "[Calibrazione" << name_suffix << "] ATTENZIONE: istogramma vuoto, nessuna stima possibile.\n";
    	}
    	c0->Update();

    	std::cout << "\n[Calibrazione" << name_suffix << " - stima robusta] eventi=" << h.total_events_processed
    	          << "  entrate=" << h.total_tracks_found
    	          << "  mu_q=" << stima.mu
    	          << "  sigma_q=" << stima.sigma << std::endl;

    	return {stima.mu, stima.sigma};
}

void Visualizer::RunNoiseScan(float mean, float std_dev, float mu_q, float sigma_q, int n_events_per_step,
                                float noise_min, float noise_max, int n_noise_steps) {
    	std::cout << "\n--- Avvio Scansione Statistica del Rumore (PUNTO 3) ---" << std::endl;
    	std::cout << "Eventi per step: " << n_events_per_step
    	          << " | Range rumore: [" << noise_min * 100.0f << "%, " << noise_max * 100.0f << "%] in "
    	          << n_noise_steps << " punti" << std::endl;
    	std::cout << "Selezione q calibrata: [" << mu_q - 3.0f * sigma_q << ", " << mu_q + 3.0f * sigma_q << "]" << std::endl;

    	std::vector<double> noise_levels;
    	for (int k = 0; k < n_noise_steps; k++) {
        	double frac = (n_noise_steps > 1) ? (double)k / (n_noise_steps - 1) : 0.0;
        	noise_levels.push_back(noise_min + frac * (noise_max - noise_min));
    	}

    	std::vector<double> noise_pct, eff_vals, fp_vals, eff_all_vals;

    	for (double noise : noise_levels) {
        	long long total_mc_trovati = 0;
        	long long total_ghosts = 0;
        	long long total_tracks_generated = 0;
        	long long total_mc_totali = 0;

        	// Same principle as the batch generation in analisi_sorgente.cxx:
        	// Source is not thread-safe (shared std::mt19937), so each OpenMP
        	// thread builds its own local Source/Detector.
        	#pragma omp parallel reduction(+:total_mc_trovati, total_ghosts, total_tracks_generated, total_mc_totali)
        	{
        		Source sorgente_locale(mean, std_dev);
        		Detector detector_locale;

        		#pragma omp for schedule(dynamic, 4)
        		for (int e = 0; e < n_events_per_step; ++e) {
            			std::vector<float> angoli = sorgente_locale.angles();
            			std::vector<std::vector<float>> y_teoriche = sorgente_locale.y_th(angoli);
            			std::vector<std::vector<float>> y_hit = sorgente_locale.Check_hits(y_teoriche);

            			std::vector<std::vector<float>> hits_lette(N_layers);

            			for(size_t i = 0; i < N_layers; i++) {
                			std::vector<uint32_t> pix = detector_locale.pos2pix(y_hit[i]);
                			sorgente_locale.MakeNoisePhysical(static_cast<float>(noise), pix);
                			hits_lette[i] = detector_locale.pix2pos(pix);
            			}

            			// Full chain: raw tracks -> clone removal (inside
            			// find_tracks_cpu_quiet) -> calibrated q-window selection.
            			std::vector<Track> tracce_pulite = find_tracks_cpu_quiet(hits_lette);
            			std::vector<Track> tracce_selezionate = FiltraTraccePerQ(tracce_pulite, mu_q, sigma_q, 3.0f);

            			// Efficiency/ghosts via MC truth: 3-sigma Mahalanobis ellipse
            			// match in (m,q) space, see truth.cxx.
            			AnalisiTracce risultati = ValutaTracce(tracce_selezionate, angoli, sourceY);

            			total_mc_trovati += risultati.mc_trovati;
            			total_ghosts += risultati.tracce_ghost;
            			// denominator: only particles in geometric acceptance
            			// (mc_trovati + mc_persi), not the generated N_tracks.
            			total_tracks_generated += risultati.mc_trovati + risultati.mc_persi;
            			// companion denominator: ALL generated particles, including
            			// those that never entered geometric acceptance.
            			total_mc_totali += risultati.mc_totali;
        		}
        	}

        	double efficienza = (total_tracks_generated > 0)
        	    ? (static_cast<double>(total_mc_trovati) / total_tracks_generated) * 100.0 : 0.0;
        	double efficienza_all_mc = (total_mc_totali > 0)
        	    ? (static_cast<double>(total_mc_trovati) / total_mc_totali) * 100.0 : 0.0;
        	double avg_fp_per_event = static_cast<double>(total_ghosts) / n_events_per_step;

        	noise_pct.push_back(noise * 100.0);
        	eff_vals.push_back(efficienza);
        	eff_all_vals.push_back(efficienza_all_mc);
        	fp_vals.push_back(avg_fp_per_event);

        	std::cout << "Rumore: " << std::fixed << std::setprecision(2) << noise * 100.0 
                  	  << "% | Eff: " << efficienza << "% | Eff (tutte le MC): " << efficienza_all_mc
                  	  << "% | FP/evento (Ghost): " << avg_fp_per_event << std::endl;
    	}

    	// Two separate panels: efficiency (%) and ghosts/event have different
    	// scales and must not share axes.
    	TCanvas* c_scan = new TCanvas("c_noise_scan", "Prestazioni vs Rumore", 200, 200, 1400, 600);
    	c_scan->Divide(2, 1);

    	c_scan->cd(1);
    	gPad->SetGrid();
    	TGraph* g_eff = new TGraph((int)noise_pct.size(), noise_pct.data(), eff_vals.data());
    	g_eff->SetTitle("Efficienza vs Rumore (cloni rimossi + selezione q); Rumore (%); Efficienza (%)");
    	g_eff->SetLineColor(kBlue);
    	g_eff->SetMarkerColor(kBlue);
    	g_eff->SetMarkerStyle(20);
    	g_eff->SetLineWidth(2);
    	g_eff->SetMinimum(0.0);
    	g_eff->SetMaximum(105.0);
    	g_eff->Draw("ALP");

    	c_scan->cd(2);
    	gPad->SetGrid();
    	TGraph* g_fp = new TGraph((int)noise_pct.size(), noise_pct.data(), fp_vals.data());
    	g_fp->SetTitle("Falsi Positivi vs Rumore (cloni rimossi + selezione q); Rumore (%); Tracce Ghost / evento");
    	g_fp->SetLineColor(kRed);
    	g_fp->SetMarkerColor(kRed);
    	g_fp->SetMarkerStyle(21);
    	g_fp->SetLineWidth(2);
    	g_fp->SetMinimum(0.0);
    	g_fp->Draw("ALP");

    	c_scan->Update();

    	// Companion canvas, same layout: left panel restates efficiency but
    	// WITHOUT excluding MC particles outside geometric acceptance from the
    	// denominator (mc_trovati / mc_totali instead of mc_trovati /
    	// (mc_trovati + mc_persi)); right panel repeats the ghosts/event curve
    	// unchanged, since it does not depend on this choice.
    	TCanvas* c_scan_all = new TCanvas("c_noise_scan_all_mc", "Prestazioni vs Rumore (tutte le tracce MC)", 220, 220, 1400, 600);
    	c_scan_all->Divide(2, 1);

    	c_scan_all->cd(1);
    	gPad->SetGrid();
    	TGraph* g_eff_all = new TGraph((int)noise_pct.size(), noise_pct.data(), eff_all_vals.data());
    	g_eff_all->SetTitle("Efficienza vs Rumore, tutte le tracce MC (incluse le perse fuori accettanza); Rumore (%); Efficienza (%)");
    	g_eff_all->SetLineColor(kBlue);
    	g_eff_all->SetMarkerColor(kBlue);
    	g_eff_all->SetMarkerStyle(20);
    	g_eff_all->SetLineWidth(2);
    	g_eff_all->SetMinimum(0.0);
    	g_eff_all->SetMaximum(105.0);
    	g_eff_all->Draw("ALP");

    	c_scan_all->cd(2);
    	gPad->SetGrid();
    	TGraph* g_fp_all = new TGraph((int)noise_pct.size(), noise_pct.data(), fp_vals.data());
    	g_fp_all->SetTitle("Falsi Positivi vs Rumore (cloni rimossi + selezione q); Rumore (%); Tracce Ghost / evento");
    	g_fp_all->SetLineColor(kRed);
    	g_fp_all->SetMarkerColor(kRed);
    	g_fp_all->SetMarkerStyle(21);
    	g_fp_all->SetLineWidth(2);
    	g_fp_all->SetMinimum(0.0);
    	g_fp_all->Draw("ALP");

    	c_scan_all->Update();
}

void Visualizer::DrawAccuracyVsTracks(const std::vector<double>& n_tracks_vals,
                                       const std::vector<double>& eff_solo_cloni,
                                       const std::vector<double>& eff_cloni_piu_q,
                                       const std::vector<double>& pur_solo_cloni,
                                       const std::vector<double>& pur_cloni_piu_q,
                                       const std::vector<double>& eff_solo_cloni_all_mc,
                                       const std::vector<double>& eff_cloni_piu_q_all_mc)
{
    	const int n = (int)n_tracks_vals.size();
    	if (n == 0) return;

    	TCanvas* c_acc = new TCanvas("c_accuracy", "Accuratezza vs Numero di Tracce", 220, 220, 1400, 600);
    	c_acc->Divide(2, 1);

    	// --- panel 1: efficiency ---
    	c_acc->cd(1);
    	gPad->SetGrid();

    	TGraph* g_eff_cloni = new TGraph(n, n_tracks_vals.data(), eff_solo_cloni.data());
    	g_eff_cloni->SetTitle("Efficienza vs Tracce generate/evento; Tracce generate per evento; Efficienza (%)");
    	g_eff_cloni->SetLineColor(kBlue);
    	g_eff_cloni->SetMarkerColor(kBlue);
    	g_eff_cloni->SetMarkerStyle(20);
    	g_eff_cloni->SetLineWidth(2);
    	g_eff_cloni->SetMinimum(0.0);
    	g_eff_cloni->SetMaximum(105.0);
    	g_eff_cloni->Draw("ALP");

    	TGraph* g_eff_q = new TGraph(n, n_tracks_vals.data(), eff_cloni_piu_q.data());
    	g_eff_q->SetLineColor(kRed);
    	g_eff_q->SetMarkerColor(kRed);
    	g_eff_q->SetMarkerStyle(21);
    	g_eff_q->SetLineWidth(2);
    	g_eff_q->Draw("LP SAME");

    	TLegend* leg1 = new TLegend(0.15, 0.15, 0.55, 0.30);
    	leg1->AddEntry(g_eff_cloni, "Solo rimozione cloni", "lp");
    	leg1->AddEntry(g_eff_q, "Cloni + selezione q calibrata", "lp");
    	leg1->Draw();

    	// --- panel 2: purity ---
    	c_acc->cd(2);
    	gPad->SetGrid();

    	TGraph* g_pur_cloni = new TGraph(n, n_tracks_vals.data(), pur_solo_cloni.data());
    	g_pur_cloni->SetTitle("Purezza vs Tracce generate/evento; Tracce generate per evento; Purezza (%)");
    	g_pur_cloni->SetLineColor(kBlue);
    	g_pur_cloni->SetMarkerColor(kBlue);
    	g_pur_cloni->SetMarkerStyle(20);
    	g_pur_cloni->SetLineWidth(2);
    	g_pur_cloni->SetMinimum(0.0);
    	g_pur_cloni->SetMaximum(105.0);
    	g_pur_cloni->Draw("ALP");

    	TGraph* g_pur_q = new TGraph(n, n_tracks_vals.data(), pur_cloni_piu_q.data());
    	g_pur_q->SetLineColor(kRed);
    	g_pur_q->SetMarkerColor(kRed);
    	g_pur_q->SetMarkerStyle(21);
    	g_pur_q->SetLineWidth(2);
    	g_pur_q->Draw("LP SAME");

    	TLegend* leg2 = new TLegend(0.15, 0.15, 0.55, 0.30);
    	leg2->AddEntry(g_pur_cloni, "Solo rimozione cloni", "lp");
    	leg2->AddEntry(g_pur_q, "Cloni + selezione q calibrata", "lp");
    	leg2->Draw();

    	c_acc->Update();

    	// Companion canvas, same layout: left panel restates efficiency but
    	// counting mc_trovati / mc_totali (NOT excluding MC particles that
    	// missed a layer and never entered geometric acceptance from the
    	// denominator); right panel repeats the same purity curves, since
    	// purity does not depend on this choice. Only drawn when the caller
    	// actually provided the companion efficiency data.
    	if (!eff_solo_cloni_all_mc.empty() && !eff_cloni_piu_q_all_mc.empty()
    	    && (int)eff_solo_cloni_all_mc.size() == n && (int)eff_cloni_piu_q_all_mc.size() == n) {

    		TCanvas* c_acc_all = new TCanvas("c_accuracy_all_mc", "Accuratezza vs Numero di Tracce (tutte le tracce MC)", 240, 240, 1400, 600);
    		c_acc_all->Divide(2, 1);

    		c_acc_all->cd(1);
    		gPad->SetGrid();

    		TGraph* g_eff_cloni_all = new TGraph(n, n_tracks_vals.data(), eff_solo_cloni_all_mc.data());
    		g_eff_cloni_all->SetTitle("Efficienza vs Tracce generate/evento, tutte le tracce MC; Tracce generate per evento; Efficienza (%)");
    		g_eff_cloni_all->SetLineColor(kBlue);
    		g_eff_cloni_all->SetMarkerColor(kBlue);
    		g_eff_cloni_all->SetMarkerStyle(20);
    		g_eff_cloni_all->SetLineWidth(2);
    		g_eff_cloni_all->SetMinimum(0.0);
    		g_eff_cloni_all->SetMaximum(105.0);
    		g_eff_cloni_all->Draw("ALP");

    		TGraph* g_eff_q_all = new TGraph(n, n_tracks_vals.data(), eff_cloni_piu_q_all_mc.data());
    		g_eff_q_all->SetLineColor(kRed);
    		g_eff_q_all->SetMarkerColor(kRed);
    		g_eff_q_all->SetMarkerStyle(21);
    		g_eff_q_all->SetLineWidth(2);
    		g_eff_q_all->Draw("LP SAME");

    		TLegend* leg1_all = new TLegend(0.15, 0.15, 0.55, 0.30);
    		leg1_all->AddEntry(g_eff_cloni_all, "Solo rimozione cloni", "lp");
    		leg1_all->AddEntry(g_eff_q_all, "Cloni + selezione q calibrata", "lp");
    		leg1_all->Draw();

    		c_acc_all->cd(2);
    		gPad->SetGrid();

    		TGraph* g_pur_cloni_all = new TGraph(n, n_tracks_vals.data(), pur_solo_cloni.data());
    		g_pur_cloni_all->SetTitle("Purezza vs Tracce generate/evento; Tracce generate per evento; Purezza (%)");
    		g_pur_cloni_all->SetLineColor(kBlue);
    		g_pur_cloni_all->SetMarkerColor(kBlue);
    		g_pur_cloni_all->SetMarkerStyle(20);
    		g_pur_cloni_all->SetLineWidth(2);
    		g_pur_cloni_all->SetMinimum(0.0);
    		g_pur_cloni_all->SetMaximum(105.0);
    		g_pur_cloni_all->Draw("ALP");

    		TGraph* g_pur_q_all = new TGraph(n, n_tracks_vals.data(), pur_cloni_piu_q.data());
    		g_pur_q_all->SetLineColor(kRed);
    		g_pur_q_all->SetMarkerColor(kRed);
    		g_pur_q_all->SetMarkerStyle(21);
    		g_pur_q_all->SetLineWidth(2);
    		g_pur_q_all->Draw("LP SAME");

    		TLegend* leg2_all = new TLegend(0.15, 0.15, 0.55, 0.30);
    		leg2_all->AddEntry(g_pur_cloni_all, "Solo rimozione cloni", "lp");
    		leg2_all->AddEntry(g_pur_q_all, "Cloni + selezione q calibrata", "lp");
    		leg2_all->Draw();

    		c_acc_all->Update();
    	}
}

void Visualizer::DrawTimingVsTracks(const std::vector<double>& n_tracks_vals,
                                     const std::vector<double>& cpu_ms_per_evento,
                                     const std::vector<double>& gpu_ms_per_evento)
{
    	const int n = (int)n_tracks_vals.size();
    	if (n == 0) return;

    	TCanvas* c_time = new TCanvas("c_timing", "Tempi di Track Finding", 240, 240, 900, 650);
    	c_time->SetGrid();

    	// With two overlaid curves the vertical range is set by the FIRST graph
    	// drawn, so set it explicitly from the global min/max of both series,
    	// otherwise the second curve can fall (partly) outside the axes.
    	double y_min = 1e30, y_max = -1e30;
    	for (double v : cpu_ms_per_evento) { y_min = std::min(y_min, v); y_max = std::max(y_max, v); }
    	for (double v : gpu_ms_per_evento) { y_min = std::min(y_min, v); y_max = std::max(y_max, v); }
    	const double y_margin = 0.10 * (y_max - y_min > 0 ? (y_max - y_min) : y_max);

    	TGraph* g_cpu = new TGraph(n, n_tracks_vals.data(), cpu_ms_per_evento.data());
    	g_cpu->SetTitle("Tempo di track finding: CPU vs GPU; Tracce generate per evento; Tempo medio per evento (ms)");
    	g_cpu->SetLineColor(kBlue);
    	g_cpu->SetMarkerColor(kBlue);
    	g_cpu->SetMarkerStyle(20);
    	g_cpu->SetLineWidth(2);
    	g_cpu->SetMinimum(std::max(0.0, y_min - y_margin));
    	g_cpu->SetMaximum(y_max + y_margin);
    	g_cpu->Draw("ALP");

    	TGraph* g_gpu = new TGraph(n, n_tracks_vals.data(), gpu_ms_per_evento.data());
    	g_gpu->SetLineColor(kGreen+2);
    	g_gpu->SetMarkerColor(kGreen+2);
    	g_gpu->SetMarkerStyle(21);
    	g_gpu->SetLineWidth(2);
    	g_gpu->Draw("LP SAME");

    	TLegend* leg = new TLegend(0.15, 0.70, 0.52, 0.87);
    	leg->SetBorderSize(1);
    	leg->SetFillStyle(1001);
    	leg->SetTextSize(0.035);
    	leg->AddEntry(g_cpu, "CPU (ricerca tracce grezze)", "lp");
    	leg->AddEntry(g_gpu, "GPU (buffer persistenti)", "lp");
    	leg->Draw();

    	c_time->Update();
}

void Visualizer::Run() {
    	m_app->Run();
}
