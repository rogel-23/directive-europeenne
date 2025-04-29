import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Interface Streamlit
st.set_page_config(page_title="Outil Directive Européenne", page_icon="📊", layout="wide")

st.markdown("## 🔒 Accès sécurisé")

password = st.text_input("Veuillez entrer le mot de passe :", type="password")

correct_password = "directive2025"

if password != correct_password:
    st.warning("⛔ Mot de passe incorrect ou vide.")
    st.stop()
else:
    st.success("🔓 Accès autorisé.")


st.title("📊 Analyse économétrique de la rémunération")

# 🧠 Tableau comparatif des modèles
st.markdown("### 🤖 Comparaison des modèles disponibles")

comparaison_modeles = pd.DataFrame({
    "Caractéristique": [
        "Relation entre variables",
        "Captation des interactions",
        "Sensibilité au bruit",
        "Interprétabilité",
        "Performance prédictive",
        "Hypothèse de linéarité",
        "Gestion des non-linéarités"
    ],
    "Régression Linéaire": [
        "Linéaire",
        "Non",
        "Élevée",
        "🌟 Très bonne",
        "⚠️ Moyenne",
        "Oui",
        "Non"
    ],
    "Arbre de Décision": [
        "Segmentaire (si/alors)",
        "Oui (via les branches)",
        "Moyenne",
        "Bonne (arbre lisible)",
        "Bonne",
        "Non",
        "Partielle"
    ],
    "Random Forest": [
        "Complexe (agrégation d'arbres)",
        "Oui (automatique)",
        "Faible",
        "❓ Moyenne (boîte noire)",
        "🌟 Très bonne",
        "Non",
        "Oui (forte)"
    ]
})

st.dataframe(comparaison_modeles.set_index("Caractéristique"))


uploaded_file = st.file_uploader("📁 Chargez votre fichier Excel ou CSV", type=["xlsx", "xlsm", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.copy()
    today = pd.to_datetime("today")
    colonnes_ajoutees = []
    col_replacements = {}

    for col in df.columns:
        col_lower = col.lower()
        if "date" not in col_lower:
            continue

        try:
            dates = pd.to_datetime(df[col], errors='coerce')
            if dates.notna().mean() < 0.5:
                continue
        except:
            continue

        base_col = col.strip()

        # Toujours créer les colonnes dérivées à partir de dates explicites
        col_days = f"{base_col} - Ancienneté (jours)"
        col_year = f"{base_col} - Année"
        col_month = f"{base_col} - Mois"
        df[col_days] = (today - dates).dt.days
        df[col_year] = dates.dt.year
        df[col_month] = dates.dt.month
        colonnes_ajoutees += [col_days, col_year, col_month]
        col_replacements[col] = col_days

        # Si c’est une date de naissance, on ajoute l’âge
        if any(k in base_col.lower() for k in ["birth", "naiss", "dob"]):
            age_col = f"{base_col} - Âge (années)"
            df[age_col] = ((today - dates).dt.days / 365.25).round().astype('Int64')
            colonnes_ajoutees.append(age_col)
            col_replacements[col] = age_col

    df.drop(columns=col_replacements.keys(), inplace=True)

    if colonnes_ajoutees:
        st.info("🛠️ Colonnes exploitables ajoutées automatiquement à partir des dates :\n- " + "\n- ".join(colonnes_ajoutees))

    st.subheader("📋 Aperçu des données")
    st.dataframe(df.head())

    all_columns = df.columns.tolist()
    date_columns = [col for col in all_columns if 'date' in col.lower()]
    non_date_columns = [col for col in all_columns if col not in date_columns]
    target = st.selectbox("🎯 Variable à expliquer (cible)", non_date_columns)


    # Calculer un R² pour chaque variable prise seule
    r2_scores = []
    for var in non_date_columns:
        if var == target:
            continue
        try:
            X_single = df[[var]].copy()
            y_single = df[[target]].copy()
            df_temp = pd.concat([X_single, y_single], axis=1).dropna()

            if df_temp.shape[0] < 5:
                continue  # Trop peu d'observations

            X_clean = pd.get_dummies(df_temp[[var]], drop_first=True)
            y_clean = df_temp[target]
            model = LinearRegression()
            model.fit(X_clean, y_clean)
            score = model.score(X_clean, y_clean)
            r2_scores.append((var, score))
        except Exception as e:
            continue


    # Trier par R² décroissant et prendre les top variables
    r2_scores.sort(key=lambda x: x[1], reverse=True)
    top_variables = [x[0] for x in r2_scores[:5]]


    features = st.multiselect(
        "📌 Variables explicatives",
        [col for col in non_date_columns if col != target],
        default=top_variables
    )

    selected_date_features = st.multiselect("📆 Variables de type date (transformées)", [col for col in date_columns if col != target])

    # On fusionne les deux listes de variables explicatives sélectionnées
    features = features + selected_date_features

    model_type = st.selectbox("🧠 Modèle à appliquer", ["Régression Linéaire", "Arbre de Décision", "Random Forest"])

    if target and features:
        X = df[features].copy()
        y = df[target]

        # Supprimer colonnes datetime
        datetime_cols = X.select_dtypes(include=["datetime", "datetime64[ns]"]).columns
        if len(datetime_cols) > 0:
            st.warning(f"🕒 Colonnes datetime ignorées : {', '.join(datetime_cols)}")
            X = X.drop(columns=datetime_cols)

        # One-hot encoding pour les colonnes catégorielles
        X = pd.get_dummies(X, drop_first=True)

        # Suppression des lignes avec NaN
        if X.isnull().any().any() or y.isnull().any():
            st.warning("⚠️ Valeurs manquantes détectées — lignes concernées supprimées.")
            df_clean = pd.concat([X, y], axis=1).dropna()
            X = df_clean[X.columns]
            y = df_clean[y.name]

        if X.shape[0] == 0:
            st.error("❌ Aucune ligne exploitable après nettoyage (valeurs manquantes). Veuillez vérifier les données chargées ou ajuster les variables sélectionnées.")
            st.stop()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        if model_type == "Régression Linéaire":
            model = LinearRegression()
        elif model_type == "Arbre de Décision":
            model = DecisionTreeRegressor(max_depth=4, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)

        st.markdown("---")
        st.markdown(
            "<div style='background-color:#e6f4ff;padding:10px;border-radius:8px;border-left:5px solid #3399ff;'>"
            "<h4>📈 Score du modèle</h4>",
            unsafe_allow_html=True
        )

        st.subheader("📈 Résultats")
        st.markdown("""
        <small>ℹ️ Le **score R²** indique la proportion de la variance de la variable cible expliquée par le modèle :
        - **R² = 1** : prédictions parfaites,
        - **R² = 0** : aucune capacité explicative,
        - **R² < 0** : pire qu’un modèle constant.
        </small>
        """, unsafe_allow_html=True)

        if score < -1 or score > 1e6:
            st.error(f"❌ Score R² incohérent : {score:.2e} — vérifiez vos données !")
        else:
            st.markdown(f"""
            <div style="background-color:#f0f8ff;padding:10px;border-radius:8px;border:1px solid #d0d0d0;">
            <b>📈 Score R² (coefficient de détermination) :</b><br>
            <span style="font-size:22px;color:#1e90ff;"><b>{score:.3f}</b></span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<div style='background-color:#fff9e6;padding:10px;border-radius:8px;border-left:5px solid #ffd24d;'>"
            "<h4>🧩 Analyse métier</h4>",
            unsafe_allow_html=True
        )

        # Analyse des variables
        if model_type == "Régression Linéaire":
            st.subheader("📊 Impact marginal estimé des variables")

            coef_values = model.coef_
            baseline = model.predict(X).mean()

            marginal_effects = []

            for var, coef in zip(X.columns, coef_values):
                if X[var].nunique() <= 10:
                    # On suppose binaire 0/1 → impact direct = coef
                    impact = coef
                else:
                    # Impact d’une variation d’un écart-type
                    std = X[var].std()
                    impact = coef * std

                marginal_effects.append((var, impact))

            # Construction DataFrame
            marginal_df = pd.DataFrame(marginal_effects, columns=["Variable", "Impact marginal (€)"])
            marginal_df["Impact_num"] = 100 * marginal_df["Impact marginal (€)"] / baseline

            # Tri et formatage
            marginal_df = marginal_df.sort_values(by="Impact_num", key=abs, ascending=False)
            marginal_df["Impact (%)"] = marginal_df["Impact_num"].map(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
            marginal_df["Impact marginal (€)"] = marginal_df["Impact marginal (€)"].map(lambda x: f"+{x:,.0f} €" if x > 0 else f"{x:,.0f} €")
            marginal_df = marginal_df.drop(columns=["Impact_num"])

            st.dataframe(marginal_df)

            st.subheader("📌 Répartition des collaborateurs")

            st.markdown("""
            <small>ℹ️ Ces graphiques montrent la **répartition de la rémunération** (variable cible) en fonction de chaque variable explicative sélectionnée.
            Ils permettent de **visualiser les relations brutes**, indépendamment du modèle.</small>
            """, unsafe_allow_html=True)

            for var in features:
                if var not in df.columns:
                    continue  # Variable supprimée / transformée

                try:
                    fig, ax = plt.subplots()
                    if df[var].nunique() <= 10:
                        # Variable catégorielle : boxplot
                        df_box = pd.concat([df[var].astype(str), df[target]], axis=1).dropna()
                        df_box.boxplot(column=target, by=var, ax=ax)
                        ax.set_title(f"{target} selon {var}")
                        ax.set_ylabel(target)
                        ax.set_xlabel(var)
                        plt.suptitle("")
                    else:
                        # Variable continue : scatterplot + droite de tendance
                        x = pd.to_numeric(df[var], errors='coerce')
                        y = pd.to_numeric(df[target], errors='coerce')
                        mask = x.notna() & y.notna()
                        x = x[mask]
                        y = y[mask]

                        ax.scatter(x, y, alpha=0.3)

                        if x.nunique() > 1:
                            coeffs = np.polyfit(x, y, deg=1)
                            poly_eq = np.poly1d(coeffs)
                            x_vals = np.linspace(x.min(), x.max(), 100)
                            y_vals = poly_eq(x_vals)
                            ax.plot(x_vals, y_vals, color='red', linestyle='--', label='Tendance linéaire')
                            ax.legend()

                        ax.set_xlabel(var)
                        ax.set_ylabel(target)
                        ax.set_title(f"{target} en fonction de {var}")

                    st.pyplot(fig)

                except Exception as e:
                    st.warning(f"⚠️ Erreur lors de la visualisation pour la variable {var} : {e}")



        else:
            st.subheader("🌟 Importance des variables")
            st.markdown("""
            <small>ℹ️ Cette mesure reflète l'**influence moyenne** de chaque variable dans les décisions du modèle.
            Plus une variable a une importance élevée, plus elle contribue aux prédictions.</small>
            """, unsafe_allow_html=True)
            # Recalibrage avec permutation + SHAP-like (impact marginal)
            st.subheader("📊 Impact marginal estimé des variables")

            # Prédiction moyenne de référence
            baseline = model.predict(X).mean()

            marginal_effects = []

            for var in X.columns:
                if X[var].nunique() <= 10:
                    # On teste la variation "0 → 1"
                    X_alt = X.copy()
                    if 0 in X[var].values and 1 in X[var].values:
                        X_alt[var] = 1
                        y_alt = model.predict(X_alt)
                        impact = y_alt.mean() - baseline
                        marginal_effects.append((var, impact))
                else:
                    # Pour variables continues : +1 std
                    X_alt = X.copy()
                    X_alt[var] += X[var].std()
                    y_alt = model.predict(X_alt)
                    impact = y_alt.mean() - baseline
                    marginal_effects.append((var, impact))

            # Création du DataFrame brut
            marginal_df = pd.DataFrame(marginal_effects, columns=["Variable", "Impact marginal (€)"])
            marginal_df["Impact_num"] = 100 * marginal_df["Impact marginal (€)"] / baseline  # colonne numérique temporaire

            # Tri sur les valeurs numériques
            marginal_df = marginal_df.sort_values(by="Impact_num", key=abs, ascending=False)

            # Formatage visuel après le tri
            marginal_df["Impact (%)"] = marginal_df["Impact_num"].map(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
            marginal_df["Impact marginal (€)"] = marginal_df["Impact marginal (€)"].map(lambda x: f"+{x:,.0f} €" if x > 0 else f"{x:,.0f} €")

            # Suppression de la colonne temporaire
            marginal_df = marginal_df.drop(columns=["Impact_num"])

            # Affichage
            st.dataframe(marginal_df)



            # ✅ Partial Dependence Plot pour Random Forest
            if model_type == "Random Forest":
                from sklearn.inspection import PartialDependenceDisplay

                st.subheader("📈 Analyse partielle de dépendance (PDP)")
                selected_pdp_var = st.selectbox("🔎 Choisissez une variable pour visualiser son effet marginal", X.columns.tolist())

                fig, ax = plt.subplots(figsize=(8, 5))
                PartialDependenceDisplay.from_estimator(model, X, [selected_pdp_var], ax=ax)
                st.pyplot(fig)

                st.markdown("""
                <small>ℹ️ Le **PDP** (Partial Dependence Plot) montre **l'effet marginal** d'une variable explicative sur la prédiction :
                autrement dit, comment la variable influence en moyenne la rémunération estimée,
                en maintenant les autres variables constantes.</small>
                """, unsafe_allow_html=True)

            if model_type == "Arbre de Décision":
                st.subheader("🌳 Arbre de Décision")
                st.markdown("""
                <small>ℹ️ Cet arbre représente les **règles de segmentation** utilisées par le modèle.
                Chaque nœud indique une **condition** sur une variable explicative, conduisant à une estimation du salaire.
                </small>
                """, unsafe_allow_html=True)
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=10, max_depth=4, ax=ax)
                st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<div style='background-color:#ffe6e6;padding:10px;border-radius:8px;border-left:5px solid #ff4d4d;'>"
            "<h4>🛠️ Contrôle qualité du modèle</h4>",
            unsafe_allow_html=True
        )


        st.subheader("📉 Résultats prédits vs réels")

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        st.markdown(f"""
        <div style="background-color:#f9f9f9;padding:10px;border-radius:8px;border:1px solid #ccc;">
        <b>📌 Résumé des performances :</b><br>
        - <b>MAE (erreur absolue moyenne)</b> : {mae:.2f}  
        - <b>RMSE (racine de l'erreur quadratique moyenne)</b> : {rmse:.2f}  
        - <b>R²</b> : {score:.3f}
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='y = x')
        ax.set_xlabel("Valeurs réelles")
        ax.set_ylabel("Valeurs prédites")
        ax.set_title("Prédiction vs Valeurs réelles")
        ax.legend()
        st.pyplot(fig)

        # Histogramme des résidus
        st.subheader("📊 Distribution des résidus")
        st.markdown("""
        <small>ℹ️ Les **résidus** correspondent à l’erreur de prédiction : `valeur réelle - valeur prédite`.
        Un bon modèle a des résidus centrés autour de 0. L’histogramme permet de visualiser la répartition de ces erreurs.
        </small>
        """, unsafe_allow_html=True)
        residuals = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.hist(residuals, bins=30, color='gray', edgecolor='black')
        ax2.set_title("Histogramme des résidus")
        ax2.set_xlabel("Erreur (y - y_pred)")
        ax2.set_ylabel("Fréquence")
        st.pyplot(fig2)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<div style='background-color:#e6ffee;padding:10px;border-radius:8px;border-left:5px solid #33cc66;'>"
            "<h4>🔍 Outil de prédiction personnalisée</h4>",
            unsafe_allow_html=True
        )

        # 🔍 Estimation personnalisée
        st.subheader("🧮 Estimation personnalisée de la rémunération")

        st.markdown("""
        <small>Complétez les caractéristiques du profil ci-dessous pour obtenir une estimation de la rémunération
        selon le modèle sélectionné et les variables explicatives retenues.</small>
        """, unsafe_allow_html=True)

        user_input = {}
        for var in features:
            if df[var].dtype == 'object' or df[var].nunique() <= 10:
                # Afficher uniquement les valeurs déjà présentes dans la base utilisée
                unique_values = df[var].dropna().unique()
                # Si mélange de types non comparables, on affiche tel quel sans tri
                try:
                    options = sorted(unique_values.tolist())
                except TypeError:
                    options = list(unique_values)

                user_input[var] = st.selectbox(f"{var}", options, key=f"user_input_{var}")

            else:
                val_min = float(df[var].min())
                val_max = float(df[var].max())
                default = float(df[var].median())
                user_input[var] = st.slider(f"{var}", int(val_min), int(val_max), int(default), step=1, key=f"user_input_{var}")

        # Transformation identique au training
        input_df = pd.DataFrame([user_input])
        input_df = pd.get_dummies(input_df)
        # Forcer à ne garder que les colonnes connues du modèle
        missing_cols = [col for col in X.columns if col not in input_df.columns]
        for col in missing_cols:
            input_df[col] = 0
        input_df = input_df[X.columns]
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        st.write("✅ Variables utilisées pour la prédiction :", input_df)

        try:
            if model_type == "Arbre de Décision":
                leaf_id = model.apply(input_df)[0]
                prediction = model.tree_.value[leaf_id][0][0]
            else:
                prediction = model.predict(input_df)[0]

            st.success(f"💰 Estimation de la variable « {target} » pour ce profil : **{prediction:.2f}**")
        except Exception as e:
            st.error("❌ Impossible de calculer la rémunération avec les données saisies.")
    
        st.markdown("</div>", unsafe_allow_html=True)
