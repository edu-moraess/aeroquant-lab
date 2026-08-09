
with tab_ml:
    methodology_block(
        info="Comparação de modelos de RUL em frota sintética.",
        method="Linear, Random Forest, GBM Quantile e MLP. Split por unidade. RMSE, MAE, NASA score.",
        interpretation="Menor RMSE = melhor ajuste. NASA penaliza superestimação. SHAP explica o modelo, não a física.",
        limitations="Só dados sintéticos. Modelos sequenciais estão na aba Neural Net.",
        label="Sobre este painel",
    )
    m1, m2, m3 = st.columns(3)
    with m1:
        ml_units = st.slider("Unidades", 12, 48, 24, 4, key="ml_u")
    with m2:
        ml_seed = st.number_input("Seed", value=2026, step=1, key="ml_s")
    with m3:
        ml_trees = st.slider("Árvores", 30, 120, 60, 10, key="ml_t")
    if st.button("Treinar", type="primary", key="ml_btn"):
        with st.spinner("Treinando..."):
            try:
                st.session_state["ml_result"] = run_ml_experiment(n_units=int(ml_units), seed=int(ml_seed), noise_std=noise_std, n_estimators=int(ml_trees))
                st.session_state.pop("shap_exp", None)
            except Exception as e:
                st.error("Falha no treino.")
                st.caption(str(e))
    if "ml_result" in st.session_state:
        res = st.session_state["ml_result"]
        st.caption(f"Treino {res.n_train_units} · Teste {res.n_test_units} · Features {res.n_features} · Melhor: {res.best_model_name}")
        st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Predito × Verdadeiro**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_best, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="Predições"))
            mx = float(max(res.test_true.max(), res.test_pred_best.max()))
            fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
            fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        with cb:
            st.markdown("**Importância**")
            if res.feature_importance is not None:
                fig = go.Figure(go.Bar(x=res.feature_importance["importance"][::-1], y=res.feature_importance["feature"][::-1], orientation="h", marker_color=THEME["SERIES_B"]))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="Importância", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            else:
                st.info("Disponível para RF e GBM.")
        residual = res.test_pred_best - res.test_true
        st.markdown("**Residual**")
        fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
        fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
        fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        st.markdown("**SHAP**")
        if res.trained_model is not None and res.X_test is not None:
            if st.button("Calcular SHAP", key="shap_btn"):
                with st.spinner("SHAP..."):
                    try:
                        st.session_state["shap_exp"] = explain_model(res.trained_model, res.X_test, max_samples=min(150, len(res.X_test)), local_index=0)
                    except Exception as e:
                        st.error("SHAP indisponível para este modelo.")
                        st.caption(str(e))
            if "shap_exp" in st.session_state:
                exp = st.session_state["shap_exp"]
                st.caption(f"{exp.method} · n={exp.n_samples_explained} · base={exp.base_value:.2f}")
                sx, sy = st.columns(2)
                with sx:
                    fig = go.Figure(go.Bar(x=exp.feature_importance["mean_abs_shap"][::-1], y=exp.feature_importance["feature"][::-1], orientation="h", marker_color=THEME["SERIES_C"]))
                    fig.update_layout(**plotly_layout(THEME, height=360, title="|SHAP| médio", x_title="mean |SHAP|", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                with sy:
                    if exp.local_shap is not None:
                        colors = [THEME["ERROR"] if v < 0 else THEME["SUCCESS"] for v in exp.local_shap["shap_value"][::-1]]
                        fig = go.Figure(go.Bar(x=exp.local_shap["shap_value"][::-1], y=exp.local_shap["feature"][::-1], orientation="h", marker_color=colors))
                        fig.update_layout(**plotly_layout(THEME, height=360, title="Local", x_title="SHAP", show_legend=False))
                        st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                    else:
                        st.info("Local requer TreeSHAP.")
        else:
            st.caption("Treine um modelo para habilitar SHAP.")
    else:
        st.info("Clique em **Treinar** para comparar modelos.")

with tab_nn:
    methodology_block(
        info="Redes neurais para RUL: MLP tabular e modelos sequenciais (janelas temporais).",
        method="MLP tabular: features engenheiradas. Sequence MLP: janelas (T×F) achatadas + MLP. LSTM/Transformer: sequências com PyTorch (se instalado). Split por unidade.",
        interpretation="Compare RMSE/MAE entre modos. Sequence models capturam dependência temporal; LSTM/Transformer exigem torch.",
        limitations="Dados sintéticos. LSTM/Transformer opcionais (torch). Preferir Sequence MLP no Cloud.",
        label="Sobre este painel",
    )

    mode = st.radio(
        "Modo",
        options=["MLP tabular", "Sequence MLP", "LSTM", "Transformer"],
        horizontal=True,
        key="nn_mode",
        help="LSTM e Transformer requerem PyTorch. No Streamlit Cloud use Sequence MLP se torch não estiver disponível.",
    )

    if mode == "MLP tabular":
        n1, n2, n3, n4 = st.columns(4)
        with n1:
            nn_units = st.slider("Unidades", 12, 40, 24, 4, key="nn_u")
        with n2:
            nn_seed = st.number_input("Seed", value=2026, step=1, key="nn_s")
        with n3:
            nn_arch = st.selectbox("Arquitetura", options=["(32,)", "(64, 32)", "(128, 64)", "(64, 32, 16)"], index=1, key="nn_a")
        with n4:
            nn_iter = st.slider("Max iterações", 50, 400, 200, 25, key="nn_i")
        compare_bl = st.checkbox("Comparar com Linear e RF", value=True, key="nn_cmp")
        if st.button("Treinar MLP", type="primary", key="nn_btn"):
            hidden = tuple(int(x.strip()) for x in nn_arch.strip("()").split(",") if x.strip())
            with st.spinner("Treinando MLP tabular..."):
                try:
                    st.session_state["nn_result"] = run_nn_experiment(
                        n_units=int(nn_units), seed=int(nn_seed), noise_std=noise_std,
                        hidden=hidden, max_iter=int(nn_iter), compare_baselines=bool(compare_bl),
                    )
                    st.session_state.pop("seq_result", None)
                except Exception as e:
                    st.error("Falha no treino da rede neural.")
                    st.caption(str(e))
        if "nn_result" in st.session_state:
            res = st.session_state["nn_result"]
            st.caption(f"{res.architecture} · épocas {res.n_epochs} · treino {res.n_train_units} · teste {res.n_test_units} · features {res.n_features}")
            st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Predito × Verdadeiro (MLP)**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred_mlp, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name="MLP"))
                mx = float(max(res.test_true.max(), res.test_pred_mlp.max()))
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with c2:
                st.markdown("**Curva de perda**")
                if res.loss_curve:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=res.loss_curve, mode="lines", line=dict(color=THEME["SERIES_B"], width=2), name="loss"))
                    fig.update_layout(**plotly_layout(THEME, height=340, x_title="Época", y_title="Loss", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                else:
                    st.info("Curva de perda indisponível.")
            residual = res.test_pred_mlp - res.test_true
            st.markdown("**Residual**")
            fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
            fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info("Configure e clique em **Treinar MLP**.")

    else:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            seq_units = st.slider("Unidades", 12, 36, 20, 4, key="seq_u")
        with s2:
            seq_len = st.slider("Janela T", 10, 50, 30, 5, key="seq_t")
        with s3:
            seq_seed = st.number_input("Seed", value=2026, step=1, key="seq_s")
        with s4:
            if mode in ("LSTM", "Transformer"):
                seq_epochs = st.slider("Épocas", 10, 80, 30, 5, key="seq_e")
            else:
                seq_iter = st.slider("Max iterações", 50, 300, 150, 25, key="seq_i")

        model_map = {"Sequence MLP": "sequence_mlp", "LSTM": "lstm", "Transformer": "transformer"}
        label = f"Treinar {mode}"
        if st.button(label, type="primary", key="seq_btn"):
            with st.spinner("Treinando modelo sequencial..."):
                try:
                    kwargs = dict(
                        n_units=int(seq_units),
                        seed=int(seq_seed),
                        noise_std=noise_std,
                        seq_len=int(seq_len),
                        stride=2,
                        model=model_map.get(mode, "sequence_mlp"),
                    )
                    if mode in ("LSTM", "Transformer"):
                        kwargs["lstm_epochs"] = int(seq_epochs)
                    else:
                        kwargs["max_iter"] = int(seq_iter)
                    st.session_state["seq_result"] = run_seq_experiment(**kwargs)
                    st.session_state.pop("nn_result", None)
                except Exception as e:
                    st.error("Falha no treino sequencial.")
                    st.caption(str(e))

        if "seq_result" in st.session_state:
            res = st.session_state["seq_result"]
            st.caption(
                f"{res.algorithm} · T={res.seq_len} · épocas {res.n_epochs} · "
                f"janelas treino {res.n_train_windows} · teste {res.n_test_windows} · "
                f"features {res.n_features} · torch={'sim' if res.torch_available else 'não'}"
            )
            st.dataframe(res.metrics_table, use_container_width=True, hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Predito × Verdadeiro**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res.test_true, y=res.test_pred, mode="markers", marker=dict(size=5, opacity=0.45, color=THEME["SERIES_A"]), name=res.model_name))
                mx = float(max(res.test_true.max(), res.test_pred.max()))
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=THEME["SERIES_MUTED"], dash="dash"), name="Ideal"))
                fig.update_layout(**plotly_layout(THEME, height=340, x_title="RUL verdadeiro", y_title="RUL previsto"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with c2:
                st.markdown("**Curva de perda**")
                if res.loss_curve:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=res.loss_curve, mode="lines", line=dict(color=THEME["SERIES_C"], width=2), name="loss"))
                    fig.update_layout(**plotly_layout(THEME, height=340, x_title="Época", y_title="Loss", show_legend=False))
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
                else:
                    st.info("Curva de perda indisponível.")
            residual = res.test_pred - res.test_true
            st.markdown("**Residual**")
            fig = go.Figure(go.Histogram(x=residual, nbinsx=36, marker_color=THEME["SERIES_D"]))
            fig.add_vline(x=0, line_dash="dot", line_color=THEME["SERIES_MUTED"])
            fig.update_layout(**plotly_layout(THEME, height=240, x_title="pred − true", show_legend=False))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
        else:
            st.info(f"Configure e clique em **{label}**.")
            if mode in ("LSTM", "Transformer"):
                st.caption(f"{mode} requer o pacote `torch`. Se falhar no Cloud, use Sequence MLP.")

with tab_anom:
    methodology_block(
        info="Detecção de anomalias em trajetórias de sensores e Health Index.",
        method="Isolation Forest: treinado nos primeiros ciclos (regime saudável). Residual z-score: salto de HI na janela local (família do Digital Twin).",
        interpretation="Taxa e scores altos perto do fim de vida ou falhas abruptas são esperados. Compare com a flag dt_anomaly do Digital Twin.",
        limitations="Sem ground-truth industrial. Contamination do IF é hiperparâmetro.",
        label="Sobre este painel",
    )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        anom_method = st.selectbox(
            "Método",
            options=["isolation_forest", "residual_zscore"],
            format_func=lambda x: "Isolation Forest" if x == "isolation_forest" else "Residual z-score",
            key="anom_m",
        )
    with a2:
        anom_units = st.slider("Unidades", 8, 32, 16, 2, key="anom_u")
    with a3:
        if anom_method == "isolation_forest":
            anom_cont = st.slider("Contamination", 0.01, 0.15, 0.05, 0.01, key="anom_c")
        else:
            anom_z = st.slider("Limiar z", 2.0, 5.0, 3.0, 0.5, key="anom_z")
    with a4:
        anom_seed = st.number_input("Seed", value=42, step=1, key="anom_s")

    if st.button("Detectar", type="primary", key="anom_btn"):
        with st.spinner("Detectando anomalias..."):
            try:
                kwargs = dict(
                    n_units=int(anom_units),
                    seed=int(anom_seed),
                    noise_std=noise_std,
                    abrupt_rate=abrupt_rate,
                    method=anom_method,
                    coupling_threshold=coupling_threshold,
                )
                if anom_method == "isolation_forest":
                    kwargs["contamination"] = float(anom_cont)
                else:
                    kwargs["z_threshold"] = float(anom_z)
                st.session_state["anom_result"] = run_anomaly_experiment(**kwargs)
            except Exception as e:
                st.error("Falha na detecção de anomalias.")
                st.caption(str(e))

    if "anom_result" in st.session_state:
        res = st.session_state["anom_result"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Amostras", res.n_samples)
        k2.metric("Anomalias", res.n_anomalies)
        k3.metric("Taxa", f"{100 * res.rate:.1f}%")
        k4.metric("Limiar", f"{res.threshold:.3f}")
        st.caption(f"Método: {res.method} · unidades: {res.n_units}")

        st.markdown("**Por unidade**")
        st.dataframe(res.by_unit, use_container_width=True, hide_index=True, height=260)

        unit_opts = res.by_unit["unit_id"].tolist()
        if unit_opts:
            sel = st.selectbox("Unidade (timeline)", options=unit_opts, index=0, key="anom_unit")
            sub = res.timeline[res.timeline["unit_id"] == sel].sort_values("cycle")
            anom = sub[sub["is_anomaly"]]

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Score de anomalia**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["score"], mode="lines", name="score", line=dict(color=THEME["SERIES_A"], width=1.5)))
                if len(anom):
                    fig.add_trace(go.Scatter(x=anom["cycle"], y=anom["score"], mode="markers", name="anomalia", marker=dict(color=THEME["ERROR"], size=7, symbol="x")))
                fig.add_hline(y=res.threshold, line_dash="dash", line_color=THEME["SERIES_MUTED"], annotation_text="limiar")
                fig.update_layout(**plotly_layout(THEME, height=320, x_title="Ciclo", y_title="Score"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with c2:
                st.markdown("**Health Index**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub["cycle"], y=sub["health_index"], mode="lines", name="HI", line=dict(color=THEME["SERIES_C"], width=2)))
                if len(anom):
                    fig.add_trace(go.Scatter(x=anom["cycle"], y=anom["health_index"], mode="markers", name="anomalia", marker=dict(color=THEME["ERROR"], size=7, symbol="x")))
                fig.update_layout(**plotly_layout(THEME, height=320, x_title="Ciclo", y_title="HI"))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)

            if "dt_anomaly" in sub.columns:
                agree = int(((sub["is_anomaly"]) & (sub["dt_anomaly"].fillna(False))).sum())
                st.caption(f"Concordância com flag do Digital Twin (mesma unidade): {agree} ciclos")
    else:
        st.info("Configure o método e clique em **Detectar**.")

with tab_mc:
    methodology_block(
        info="Propagação de incerteza do RUL.",
        method="Aleatória: ruído/seed, limiar fixo. Epistêmica: trajetória fixa, percentil do limiar (30–70).",
        interpretation="Distribuição do RUL e variâncias aproximam H3. Não é Bayesiana formal.",
        limitations="Fontes não ortogonais. Custo cresce com n_runs.",
        label="Sobre este painel",
    )
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        mc_runs = st.slider("Trajetórias", 8, 40, 16, 2, key="mc_r")
    with mc2:
        mc_life = st.slider("Vida útil", 80, 220, 140, 10, key="mc_l")
    with mc3:
        mc_frac = st.slider("Fração de vida", 0.4, 0.8, 0.6, 0.05, key="mc_f")
    with mc4:
        mc_seed = st.number_input("Seed", value=42, step=1, key="mc_s")
    if st.button("Executar", type="primary", key="mc_btn"):
        with st.spinner("Monte Carlo..."):
            try:
                st.session_state["mc_result"] = run_monte_carlo_rul(
                    n_runs=int(mc_runs), max_cycles=int(mc_life), reference_cycle_fraction=float(mc_frac),
                    base_seed=int(mc_seed), noise_std=noise_std, n_calibration_units=8)
            except Exception as e:
                st.error("Falha no Monte Carlo.")
                st.caption(str(e))
    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        if r.n_runs < 1 or not np.isfinite(r.mean):
            st.warning("Dados insuficientes para a distribuição de RUL.")
        else:
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("RUL médio", f"{r.mean:.1f}")
            k2.metric("Desvio", f"{r.std:.1f}")
            k3.metric("Q05–Q95", f"{r.q05:.0f}–{r.q95:.0f}")
            k4.metric("Verdadeiro", f"{r.true_rul_at_ref:.0f}")
            k5.metric("Runs", r.n_runs)
            xa, xb = st.columns(2)
            with xa:
                st.markdown("**Distribuição**")
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=r.rul_samples, nbinsx=18, marker_color=THEME["SERIES_A"]))
                fig.add_vline(x=r.true_rul_at_ref, line_dash="dash", line_color=THEME["ERROR"], annotation_text="verdadeiro", annotation_font_color=THEME["TEXT_SECONDARY"])
                fig.add_vline(x=r.mean, line_dash="solid", line_color=THEME["SERIES_B"], annotation_text="média", annotation_font_color=THEME["TEXT_SECONDARY"])
                fig.update_layout(**plotly_layout(THEME, height=320, x_title="RUL (ciclos)", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            with xb:
                st.markdown("**Variância**")
                fig = go.Figure(go.Bar(x=["Total", "Aleatória", "Epistêmica"], y=[r.var_total, r.var_aleatoric, r.var_epistemic],
                                       marker_color=[THEME["SERIES_MUTED"], THEME["SERIES_A"], THEME["SERIES_B"]]))
                fig.update_layout(**plotly_layout(THEME, height=320, y_title="Variância", show_legend=False))
                st.plotly_chart(fig, use_container_width=True, theme="streamlit", config=PLOTLY_CONFIG)
            tot = max(r.var_total, 1e-12)
            st.caption(f"Total {r.var_total:.1f} · Aleatória {r.var_aleatoric:.1f} ({100*r.var_aleatoric/tot:.0f}%) · Epistêmica {r.var_epistemic:.1f} ({100*r.var_epistemic/tot:.0f}%)")
    else:
        st.info("Configure e clique em **Executar**.")
