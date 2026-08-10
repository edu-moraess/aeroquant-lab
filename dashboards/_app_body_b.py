
with tab_ml:
    from cmapss_experiment import list_available_subsets, run_cmapss_experiment

    methodology_block(
        info="Comparação de modelos de RUL — sintético ou NASA C-MAPSS real.",
        method="Linear, RF, MLP. Split oficial C-MAPSS ou split por unit_id no sintético. Normalize fit só no treino.",
        interpretation="Ranking NASA-first. C-MAPSS é benchmark NASA (simulação), não telemetria comercial.",
        limitations="C-MAPSS requer arquivos em data/external/ (python scripts/download_cmapss.py).",
        label="Sobre este painel",
    )
    data_src = st.radio("Fonte de dados", options=["Sintético", "C-MAPSS real"], horizontal=True, key="ml_data_src")
    m1, m2, m3 = st.columns(3)
    if data_src == "C-MAPSS real":
        avail = list_available_subsets()
        with m1:
            subset = st.selectbox("Subset", options=avail or ["FD001"], key="ml_fd")
        with m2:
            ml_seed = st.number_input("Seed", value=42, step=1, key="ml_s")
        with m3:
            ml_trees = st.slider("Árvores", 30, 150, 80, 10, key="ml_t")
        if not avail:
            st.warning("Arquivos C-MAPSS não encontrados. Rode: `python scripts/download_cmapss.py`")
        btn_label = "Treinar C-MAPSS"
    else:
        with m1:
            ml_units = st.slider("Unidades", 12, 48, 24, 4, key="ml_u")
        with m2:
            ml_seed = st.number_input("Seed", value=2026, step=1, key="ml_s")
        with m3:
            ml_trees = st.slider("Árvores", 30, 120, 60, 10, key="ml_t")
        btn_label = "Treinar"
        subset = None

    if st.button(btn_label, type="primary", key="ml_btn"):
        with st.spinner("Treinando..."):
            try:
                if data_src == "C-MAPSS real":
                    st.session_state["ml_result"] = run_cmapss_experiment(
                        subset=str(subset), seed=int(ml_seed), n_estimators=int(ml_trees)
                    )
                else:
                    st.session_state["ml_result"] = run_ml_experiment(
                        n_units=int(ml_units), seed=int(ml_seed), noise_std=noise_std, n_estimators=int(ml_trees)
                    )
                st.session_state.pop("shap_exp", None)
            except Exception as e:
                st.error("Falha no treino.")
                st.caption(str(e))

    if "ml_result" in st.session_state:
        res = st.session_state["ml_result"]
        src_label = getattr(res, "subset", None) or getattr(res, "data_source", "synthetic")
        st.caption(f"{src_label} · Treino {res.n_train_units} · Teste {res.n_test_units} · Features {res.n_features} · Melhor: {res.best_model_name}")
        table = getattr(res, "ranked_table", None)
        st.dataframe(table if table is not None else res.metrics_table, width='stretch', hide_index=True)
        if getattr(res, "protocol_note", None):
            st.caption(res.protocol_note)
        if getattr(res, "bucket_table", None) is not None:
            st.markdown("**Desempenho por faixa de RUL**")
            st.dataframe(res.bucket_table, width='stretch', hide_index=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Predicted vs Actual RUL**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_best, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="Predições"))
            mx = float(max(res.test_true.max(), res.test_pred_best.max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL", y_title="Predicted RUL"))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        with cb:
            st.markdown("**Importância**")
            if getattr(res, "feature_importance", None) is not None:
                fig = go.Figure(go.Bar(x=res.feature_importance["importance"][::-1], y=res.feature_importance["feature"][::-1], orientation="h", marker_color=THEME["SERIES_B"]))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="Importância", show_legend=False))
                st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
            else:
                st.info("Disponível no modo sintético (RF/GBM).")
        residual = res.test_pred_best - res.test_true
        if getattr(res, "residual_report", None) is not None:
            st.info(res.residual_report.bias_message)
            st.dataframe(res.residual_report.summary_table(), width='stretch', hide_index=True)
        st.markdown("**Residual diagnostics**")
        rc1, rc2 = st.columns(2)
        with rc1:
            fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
            fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=260, title="Residual Distribution", x_title="pred − true", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        with rc2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_pred_best, y=residual, mode="markers", marker=dict(size=5, opacity=0.4, color=THEME["SERIES_A"])))
            fig.add_hline(y=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=260, title="Residual vs Predicted", x_title="Predicted RUL", y_title="Residual", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        rc3, rc4 = st.columns(2)
        with rc3:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=residual, mode="markers", marker=dict(size=5, opacity=0.4, color=THEME["SERIES_C"])))
            fig.add_hline(y=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=260, title="Residual vs True RUL", x_title="True RUL", y_title="Residual", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        with rc4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=np.abs(residual), mode="markers", marker=dict(size=5, opacity=0.4, color=THEME["SERIES_B"])))
            fig.update_layout(**plotly_layout(THEME, height=260, title="Absolute Error vs True RUL", x_title="True RUL", y_title="|error|", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        st.markdown("**SHAP**")
        if getattr(res, "trained_model", None) is not None and getattr(res, "X_test", None) is not None:
            if st.button("Calcular SHAP", key="shap_btn"):
                with st.spinner("SHAP..."):
                    try:
                        st.session_state["shap_exp"] = explain_model(res.trained_model, res.X_test, max_samples=min(150, len(res.X_test)), local_index=0)
                    except Exception as e:
                        st.error("SHAP indisponível."); st.caption(str(e))
            if "shap_exp" in st.session_state:
                exp = st.session_state["shap_exp"]
                st.caption(f"{exp.method} · n={exp.n_samples_explained}")
        else:
            st.caption("SHAP disponível no modo sintético com modelo treinado.")
    else:
        st.info("Escolha a fonte e clique em treinar.")

with tab_nn:
    methodology_block(
        info="Redes neurais para RUL: MLP tabular e modelos sequenciais.",
        method="MLP / Sequence MLP / LSTM / Transformer. Sequence length = últimos T ciclos.",
        interpretation="Sequence Length = 30 → últimos 30 ciclos para o RUL atual.",
        limitations="Por padrão ainda sintético. LSTM/Transformer exigem torch.",
        label="Sobre este painel",
    )
    mode = st.radio("Modo", options=["MLP tabular", "Sequence MLP", "LSTM", "Transformer"], horizontal=True, key="nn_mode")
    if mode == "MLP tabular":
        n1, n2, n3, n4 = st.columns(4)
        with n1: nn_units = st.slider("Unidades", 12, 40, 24, 4, key="nn_u")
        with n2: nn_seed = st.number_input("Seed", value=2026, step=1, key="nn_s")
        with n3: nn_arch = st.selectbox("Arquitetura", options=["(32,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"], index=1, key="nn_a")
        with n4: nn_iter = st.slider("Max iterações", 50, 400, 200, 25, key="nn_i")
        compare_bl = st.checkbox("Comparar com Linear e RF", value=True, key="nn_cmp")
        if st.button("Treinar MLP", type="primary", key="nn_btn"):
            hidden = tuple(int(x.strip()) for x in nn_arch.strip("()").split(",") if x.strip())
            with st.spinner("Treinando..."):
                try:
                    st.session_state["nn_result"] = run_nn_experiment(n_units=int(nn_units), seed=int(nn_seed), noise_std=noise_std, hidden=hidden, max_iter=int(nn_iter), compare_baselines=bool(compare_bl))
                    st.session_state.pop("seq_result", None)
                except Exception as e:
                    st.error("Falha."); st.caption(str(e))
        if "nn_result" in st.session_state:
            res = st.session_state["nn_result"]
            st.caption(f"{res.architecture} · épocas {res.n_epochs}")
            st.dataframe(res.metrics_table, width='stretch', hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_mlp, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"])))
            mx = float(max(res.test_true.max(), res.test_pred_mlp.max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash")))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL", y_title="Predicted RUL", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info("Configure e treine o MLP.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1: seq_units = st.slider("Unidades", 12, 36, 20, 4, key="seq_u")
        with s2: seq_len = st.slider("Janela T", 10, 50, 30, 5, key="seq_t")
        with s3: seq_seed = st.number_input("Seed", value=2026, step=1, key="seq_s")
        with s4:
            if mode in ("LSTM", "Transformer"): seq_epochs = st.slider("Épocas", 10, 80, 30, 5, key="seq_e")
            else: seq_iter = st.slider("Max iterações", 50, 300, 150, 25, key="seq_i")
        model_map = {"Sequence MLP": "sequence_mlp", "LSTM": "lstm", "Transformer": "transformer"}
        label = f"Treinar {mode}"
        if st.button(label, type="primary", key="seq_btn"):
            with st.spinner("Treinando..."):
                try:
                    kwargs = dict(n_units=int(seq_units), seed=int(seq_seed), noise_std=noise_std, seq_len=int(seq_len), stride=2, model=model_map.get(mode, "sequence_mlp"))
                    if mode in ("LSTM", "Transformer"): kwargs["lstm_epochs"] = int(seq_epochs)
                    else: kwargs["max_iter"] = int(seq_iter)
                    st.session_state["seq_result"] = run_seq_experiment(**kwargs)
                    st.session_state.pop("nn_result", None)
                except Exception as e:
                    st.error("Falha."); st.caption(str(e))
        if "seq_result" in st.session_state:
            res = st.session_state["seq_result"]
            st.caption(f"{res.algorithm} · T={res.seq_len}")
            st.dataframe(res.metrics_table, width='stretch', hide_index=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"])))
            mx = float(max(res.test_true.max(), res.test_pred.max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash")))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="True RUL", y_title="Predicted RUL", show_legend=False))
            st.plotly_chart(fig, width='stretch', theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(f"Configure e clique em **{label}**.")

with tab_anom:
    methodology_block(info="Detecção de anomalias.", method="Isolation Forest ou residual z-score.", interpretation="Scores altos perto do fim de vida são esperados.", limitations="Sem ground-truth industrial.", label="Sobre este painel")
    a1, a2, a3, a4 = st.columns(4)
    with a1: anom_method = st.selectbox("Método", options=["isolation_forest", "residual_zscore"], format_func=lambda x: "Isolation Forest" if x == "isolation_forest" else "Residual z-score", key="anom_m")
    with a2: anom_units = st.slider("Unidades", 8, 32, 16, 2, key="anom_u")
    with a3:
        if anom_method == "isolation_forest": anom_cont = st.slider("Contamination", 0.01, 0.15, 0.05, 0.01, key="anom_c")
        else: anom_z = st.slider("Limiar z", 2.0, 5.0, 3.0, 0.5, key="anom_z")
    with a4: anom_seed = st.number_input("Seed", value=42, step=1, key="anom_s")
    if st.button("Detectar", type="primary", key="anom_btn"):
        with st.spinner("Detectando..."):
            try:
                kwargs = dict(n_units=int(anom_units), seed=int(anom_seed), noise_std=noise_std, abrupt_rate=abrupt_rate, method=anom_method, coupling_threshold=coupling_threshold)
                if anom_method == "isolation_forest": kwargs["contamination"] = float(anom_cont)
                else: kwargs["z_threshold"] = float(anom_z)
                st.session_state["anom_result"] = run_anomaly_experiment(**kwargs)
            except Exception as e:
                st.error("Falha."); st.caption(str(e))
    if "anom_result" in st.session_state:
        res = st.session_state["anom_result"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Amostras", res.n_samples); k2.metric("Anomalias", res.n_anomalies)
        k3.metric("Taxa", f"{100 * res.rate:.1f}%"); k4.metric("Limiar", f"{res.threshold:.3f}")
        st.dataframe(res.by_unit, width='stretch', hide_index=True, height=260)
    else:
        st.info("Configure e clique em **Detectar**.")

with tab_mc:
    methodology_block(info="Incerteza de RUL + risk.", method="Monte Carlo no Digital Twin.", interpretation="Thresholds configuráveis.", limitations="Dados sintéticos no MC atual.", label="Sobre este painel")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1: mc_runs = st.slider("Trajetórias", 8, 40, 16, 2, key="mc_r")
    with mc2: mc_life = st.slider("Vida útil", 80, 220, 140, 10, key="mc_l")
    with mc3: mc_frac = st.slider("Fração de vida", 0.4, 0.8, 0.6, 0.05, key="mc_f")
    with mc4: mc_seed = st.number_input("Seed", value=42, step=1, key="mc_s")
    maint_thr = st.slider("Maintenance threshold (ciclos)", 10, 80, 30, 5, key="mc_thr")
    if st.button("Executar", type="primary", key="mc_btn"):
        with st.spinner("Monte Carlo..."):
            try:
                st.session_state["mc_result"] = run_monte_carlo_rul(n_runs=int(mc_runs), max_cycles=int(mc_life), reference_cycle_fraction=float(mc_frac), base_seed=int(mc_seed), noise_std=noise_std, n_calibration_units=8)
            except Exception as e:
                st.error("Falha."); st.caption(str(e))
    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        if r.n_runs >= 1 and np.isfinite(r.mean):
            from aeroquant.risk.assessment import assess_risk
            risk = assess_risk(r.rul_samples, maintenance_threshold=float(maint_thr))
            p10 = float(np.percentile(r.rul_samples, 10)); p90 = float(np.percentile(r.rul_samples, 90))
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Expected RUL", f"{r.mean:.1f}"); k2.metric("P10", f"{p10:.0f}"); k3.metric("P50", f"{r.q50:.0f}")
            k4.metric("P90", f"{p90:.0f}"); k5.metric(f"P(RUL<{int(maint_thr)})", f"{100*risk.prob_below_threshold:.0f}%"); k6.metric("Risk", risk.level)
            st.caption(risk.rationale)
    else:
        st.info("Configure e clique em **Executar**.")

with tab_risk:
    from risk_panel import render_risk_tab
    render_risk_tab(noise_std=noise_std, THEME=THEME, plotly_layout=plotly_layout, methodology_block=methodology_block, PLOTLY_CONFIG=PLOTLY_CONFIG)
