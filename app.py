import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Cluster de Gènes",
    page_icon="🧬",
    layout="wide"
)

# Style CSS personnalisé amélioré
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d1e3f0 100%);
        padding: 30px;
        border-radius: 20px;
        margin-top: 25px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .feature-section {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        border-left: 4px solid #1f77b4;
    }
    .cluster-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        background-color: #1f77b4;
        color: white;
        font-size: 28px;
        font-weight: bold;
        margin: 15px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        font-size: 14px;
    }
    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s;
    }
    .debug-section {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 20px 0;
        font-family: monospace;
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧬 Prédiction de Cluster de Gènes</h1>', unsafe_allow_html=True)

# ========== DÉFINITIONS DES CLUSTERS ==========
CLUSTER_PROFILES = {
    0: {
        "name": "Gènes Codants Standards",
        "description": "Gènes codants pour des protéines avec descriptions modérées, répartis uniformément sur le génome",
        "characteristics": [
            "🧬 Majoritairement des gènes protein-coding",
            "📏 Descriptions de longueur moyenne (200-400 caractères)",
            "📍 Distribution équilibrée sur tous les chromosomes",
            "⚙️ Fonctions cellulaires générales et métabolisme"
        ],
        "examples": [
            "ACTB - Actine beta (cytosquelette)",
            "GAPDH - Glycéraldéhyde-3-phosphate déshydrogénase",
            "TUBB - Tubuline beta"
        ],
        "biological_role": "Fonctions housekeeping et métabolisme de base",
        "color": "#4CAF50"
    },
    1: {
        "name": "ARN Non-Codants Régulateurs",
        "description": "ARN non-codants (lncRNA, miRNA) avec descriptions courtes, rôles régulateurs",
        "characteristics": [
            "🎭 Principalement lncRNA et miRNA",
            "📝 Descriptions très courtes (<150 caractères)",
            "🎯 Localisations chromosomiques spécifiques",
            "🔧 Régulation de l'expression génique"
        ],
        "examples": [
            "XIST - Inactivation du chromosome X",
            "H19 - ARN long non-codant imprinté",
            "MALAT1 - Régulation de la transcription"
        ],
        "biological_role": "Régulation épigénétique et contrôle post-transcriptionnel",
        "color": "#FF9800"
    },
    2: {
        "name": "Gènes Richement Annotés",
        "description": "Gènes avec descriptions très détaillées, souvent liés à des maladies",
        "characteristics": [
            "📚 Descriptions très longues (>600 caractères)",
            "🏥 Forte association avec pathologies humaines",
            "🔬 Gènes très étudiés et documentés",
            "💊 Cibles thérapeutiques potentielles"
        ],
        "examples": [
            "TP53 - Suppresseur de tumeur (cancer)",
            "BRCA1 - Cancer du sein héréditaire",
            "CFTR - Fibrose kystique"
        ],
        "biological_role": "Gènes cliniquement importants et cibles médicamenteuses",
        "color": "#E91E63"
    },
    3: {
        "name": "Pseudogènes",
        "description": "Copies non fonctionnelles de gènes, descriptions minimalistes",
        "characteristics": [
            "🚫 Pseudogènes (gènes désactivés)",
            "📉 Descriptions très courtes ou absentes",
            "🧬 Dérivés de duplications génomiques",
            "🔇 Non traduits en protéines fonctionnelles"
        ],
        "examples": [
            "PTENP1 - Pseudogène de PTEN",
            "PGAM1P - Pseudogène de phosphoglycérate mutase"
        ],
        "biological_role": "Vestiges évolutifs, potentiels régulateurs par compétition d'ARN",
        "color": "#9E9E9E"
    },
    4: {
        "name": "ARN Structuraux (rRNA, tRNA)",
        "description": "ARN essentiels à la machinerie cellulaire",
        "characteristics": [
            "⚙️ rRNA, tRNA, snRNA, snoRNA",
            "🏭 Composants de la traduction et épissage",
            "📍 Souvent en clusters génomiques",
            "🔄 Expression constitutive élevée"
        ],
        "examples": [
            "RN7SL1 - Composant de la particule SRP",
            "RMRP - ARN de la RNase MRP",
            "Gènes tRNA dispersés"
        ],
        "biological_role": "Machinerie fondamentale de synthèse protéique",
        "color": "#3F51B5"
    },
    5: {
        "name": "Gènes du Chromosome X",
        "description": "Gènes spécifiquement concentrés sur le chromosome X",
        "characteristics": [
            "❌ Localisation exclusive chromosome X",
            "👥 Liés à l'hérédité liée au sexe",
            "🧬 Soumis à l'inactivation du X (femmes)",
            "🔬 Importants pour maladies récessives liées à l'X"
        ],
        "examples": [
            "DMD - Dystrophine (dystrophie musculaire)",
            "F8 - Facteur VIII (hémophilie A)",
            "GLA - Alpha-galactosidase (maladie de Fabry)"
        ],
        "biological_role": "Pathologies liées au sexe et dosage génique",
        "color": "#9C27B0"
    },
    6: {
        "name": "Régions Biologiques",
        "description": "Éléments régulateurs et régions non géniques",
        "characteristics": [
            "🎚️ Enhancers, promoteurs, régions régulatrices",
            "📍 Ne codent pas de produits finaux",
            "🔀 Contrôle de l'expression génique à distance",
            "🧬 Importance en génétique des maladies complexes"
        ],
        "examples": [
            "Régions enhancers de gènes développementaux",
            "Promoteurs alternatifs",
            "Îlots CpG régulateurs"
        ],
        "biological_role": "Architecture régulatrice du génome",
        "color": "#00BCD4"
    },
    7: {
        "name": "Gènes Mitochondriaux",
        "description": "Gènes du génome mitochondrial",
        "characteristics": [
            "⚡ ADN mitochondrial (chromosome MT)",
            "🔋 Métabolisme énergétique",
            "👪 Hérédité maternelle exclusive",
            "🧬 37 gènes seulement (génome très compact)"
        ],
        "examples": [
            "MT-CO1 - Cytochrome c oxydase sous-unité 1",
            "MT-ND1 - NADH déshydrogénase sous-unité 1",
            "MT-ATP6 - ATP synthase sous-unité 6"
        ],
        "biological_role": "Production d'énergie cellulaire (chaîne respiratoire)",
        "color": "#FFEB3B"
    },
    8: {
        "name": "Gènes Spécialisés Rares",
        "description": "Gènes atypiques ou peu caractérisés",
        "characteristics": [
            "❓ Fonctions peu connues ou uniques",
            "🔬 Faible représentation dans les bases de données",
            "🧬 Types géniques rares (scRNA, autres)",
            "📊 Nécessitent plus de recherche"
        ],
        "examples": [
            "Gènes de familles multigéniques spécialisées",
            "Nouveaux types d'ARN non-codants"
        ],
        "biological_role": "Fonctions spécialisées ou émergentes",
        "color": "#795548"
    }
}

# Détection automatique du dossier des modèles
def find_model_dir():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_dirs = [
        os.path.join(current_script_dir, "model_final"),
        os.path.join(current_script_dir, "projet_bio_info", "model_final"),
        current_script_dir
    ]
    
    for directory in possible_dirs:
        scaler_path = os.path.join(directory, "scaler.pkl")
        if os.path.exists(scaler_path):
            return directory
    return None

model_dir = find_model_dir()

if model_dir is None:
    st.error("❌ Dossier des modèles introuvable")
    st.info("""
    Structure de dossier attendue :
    ```
    votre_projet/
    ├── app.py
    └── model_final/
        ├── scaler.pkl
        ├── kmeans_model.pkl
        └── categorical_maps.pkl
    ```
    """)
    st.stop()

# Charger les modèles
@st.cache_resource
def load_models(model_dir):
    try:
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        kmeans_path = os.path.join(model_dir, "kmeans_model.pkl")
        categorical_maps_path = os.path.join(model_dir, "categorical_maps.pkl")
        
        scaler = joblib.load(scaler_path)
        kmeans = joblib.load(kmeans_path)
        categorical_maps = joblib.load(categorical_maps_path)
        
        if not hasattr(scaler, 'scale_') or not hasattr(kmeans, 'cluster_centers_'):
            raise ValueError("Modèles corrompus ou incomplets")
            
        return scaler, kmeans, categorical_maps
        
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles : {str(e)}")
        return None, None, None

scaler, kmeans, categorical_maps = load_models(model_dir)

if scaler is None or kmeans is None or categorical_maps is None:
    st.stop()

# Mapping des types de gènes
TYPE_TO_COLUMN = {
    "protein-coding": "type_protein-coding",
    "pseudogene": "type_pseudo",
    "lncRNA": "type_ncRNA",
    "miRNA": "type_ncRNA",
    "snRNA": "type_snRNA",
    "snoRNA": "type_snoRNA",
    "rRNA": "type_rRNA",
    "scRNA": "type_scRNA",
    "tRNA": "type_tRNA",
    "ncRNA": "type_ncRNA",
    "other": "type_other",
    "biological-region": "type_biological-region"
}

# Liste complète des chromosomes (1-22 + X, Y, MT)
ALL_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']

# Obtenir les chromosomes disponibles dans le mapping (si existant)
chromosomes = ALL_CHROMOSOMES  # Toujours utiliser la liste complète

# Bras chromosomiques
arms = ['p', 'q']

gene_types = sorted(TYPE_TO_COLUMN.keys())

# ========== GUIDE DES CLUSTERS EN HAUT ==========
st.markdown("## 📊 Guide des Clusters")
st.markdown("Voici les 9 types de clusters identifiés dans notre modèle. Chaque cluster regroupe des gènes avec des caractéristiques similaires.")

# Afficher tous les clusters dans une grille
cols_per_row = 3
cluster_ids = sorted(CLUSTER_PROFILES.keys())

for i in range(0, len(cluster_ids), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        cluster_idx = i + j
        if cluster_idx < len(cluster_ids):
            cluster_id = cluster_ids[cluster_idx]
            profile = CLUSTER_PROFILES[cluster_id]
            
            with col:
                with st.expander(f"**Cluster {cluster_id}**: {profile['name']}", expanded=False):
                    st.markdown(f"<div style='color: {profile['color']}; font-weight: bold;'>{profile['description']}</div>", unsafe_allow_html=True)
                    st.markdown("**Caractéristiques principales:**")
                    for char in profile['characteristics'][:2]:
                        st.markdown(f"- {char}")

st.markdown("---")

# ========== FORMULAIRE DE PRÉDICTION ==========
st.markdown("## 🔬 Caractéristiques du gène à analyser")
st.markdown("Entrez les informations du gène pour déterminer à quel cluster il appartient.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Description")
        desc_length = st.number_input(
            "Longueur (caractères)",
            min_value=0,
            max_value=10000,
            value=250,
            step=10,
            help="Nombre de caractères dans la description fonctionnelle"
        )
    
    with col2:
        st.markdown("### 📍 Localisation")
        chromosome = st.selectbox(
            "Chromosome",
            options=chromosomes,
            index=0
        )
        arm = st.selectbox(
            "Bras (p/q)",
            options=arms,
            index=0
        )
    
    with col3:
        st.markdown("### 🧬 Type de gène")
        gene_type = st.selectbox(
            "Type",
            options=gene_types,
            index=0
        )
    
    st.markdown("")
    submitted = st.form_submit_button("🔮 Prédire le cluster", type="primary", use_container_width=True)

# ========== MODE DEBUG ==========
debug_mode = st.sidebar.checkbox("🐛 Mode Debug", value=False, help="Afficher les détails techniques du processus")

# Traitement de la prédiction
if submitted:
    try:
        # ORDRE EXACT DES 19 FEATURES UTILISÉES LORS DE L'ENTRAÎNEMENT
        COLUMNS_ORDER = [
            'GeneID', 'desc_length', 'chromosome', 'arm_encoded', 'chrom_encoded',
            'type_biological-region', 'type_ncRNA', 'type_other', 'type_protein-coding', 'type_pseudo',
            'type_rRNA', 'type_scRNA', 'type_snRNA', 'type_snoRNA', 'type_tRNA', 'type_unknown',
            'Symbol', 'type_of_gene', 'description'
        ]
        
        # Création des features avec TOUTES les 19 colonnes
        features = {}
        
        # 1. Colonnes numériques simples
        features['GeneID'] = 0.0  # Valeur arbitraire mais constante
        features['desc_length'] = float(desc_length)
        
        # 2. Colonnes catégorielles (seront encodées après)
        features['chromosome'] = str(chromosome)
        features['arm_encoded'] = str(arm)
        features['chrom_encoded'] = str(chromosome)  # Redondant mais requis
        features['Symbol'] = "PREDICTED_GENE"
        features['type_of_gene'] = str(gene_type)
        features['description'] = "Predicted gene description"
        
        # 3. One-hot encoding des types de gènes
        type_cols = [col for col in COLUMNS_ORDER if col.startswith('type_')]
        for col in type_cols:
            features[col] = 0.0
        
        if gene_type in TYPE_TO_COLUMN:
            target_col = TYPE_TO_COLUMN[gene_type]
            if target_col in type_cols:
                features[target_col] = 1.0
            else:
                features['type_unknown'] = 1.0
        else:
            features['type_unknown'] = 1.0
        
        # Création du DataFrame initial
        features_df = pd.DataFrame([features], columns=COLUMNS_ORDER)
        
        if debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.markdown("### 🐛 DEBUG - Étape 1: Features brutes (avant encodage)")
            st.dataframe(features_df.T, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Encodage robuste des variables catégorielles
        encoding_log = []
        
        for col in ['chromosome', 'arm_encoded', 'chrom_encoded', 'Symbol', 'type_of_gene', 'description']:
            if col in features_df.columns:
                original_value = str(features_df[col].iloc[0])
                encoded_value = None
                
                # 1. Essayer d'abord avec le mapping appris SI disponible ET SI la valeur est dans le mapping
                if col in categorical_maps and categorical_maps[col]:
                    mapping_dict = categorical_maps[col]
                    
                    # Essayer différentes variantes pour trouver une correspondance
                    candidates = [
                        original_value,
                        original_value.upper(),
                        original_value.lower(),
                        original_value.strip(),
                        original_value.replace('-', '').replace("'", '')
                    ]
                    
                    for candidate in candidates:
                        if candidate in mapping_dict:
                            raw_encoded = mapping_dict[candidate]
                            try:
                                encoded_value = float(raw_encoded)
                                encoding_log.append(f"✅ {col}: '{original_value}' → {encoded_value} (mapping appris)")
                                break
                            except (ValueError, TypeError):
                                # Si la valeur du mapping n'est pas convertible, ignorer et continuer
                                continue
                
                # 2. Si pas de mapping ou valeur non trouvée, utiliser encodage manuel selon le type de colonne
                if encoded_value is None:
                    if col in ['chromosome', 'chrom_encoded']:
                        # Mapping manuel complet et robuste pour les chromosomes
                        chrom_map = {str(i): float(i) for i in range(1, 23)}
                        chrom_map.update({
                            'X': 23.0, 'x': 23.0,
                            'Y': 24.0, 'y': 24.0,
                            'MT': 25.0, 'Mt': 25.0, 'mt': 25.0, 'M': 25.0, 'm': 25.0
                        })
                        # Nettoyer la valeur d'entrée
                        clean_val = original_value.strip().upper().replace('CHR', '').replace('CHROMOSOME', '')
                        encoded_value = chrom_map.get(clean_val, 1.0)  # 1.0 comme valeur par défaut sûre
                        encoding_log.append(f"🔧 {col}: '{original_value}' → {encoded_value} (mapping manuel)")
                    
                    elif col == 'arm_encoded':
                        clean_val = original_value.strip().lower()
                        encoded_value = 0.0 if clean_val in ['p', 'short', 'petit'] else 1.0
                        encoding_log.append(f"🔧 {col}: '{original_value}' → {encoded_value} (encodage manuel p=0/q=1)")
                    
                    else:
                        # Pour Symbol, type_of_gene, description : utiliser 0.0 comme fallback numérique sûr
                        encoded_value = 0.0
                        encoding_log.append(f"🔧 {col}: '{original_value}' → 0.0 (fallback numérique)")
                
                # Appliquer la valeur encodée
                features_df[col] = float(encoded_value)
        
        if debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.markdown("### 🐛 DEBUG - Étape 2: Log d'encodage")
            for log in encoding_log:
                st.text(log)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Vérification CRITIQUE de la dimension
        if features_df.shape[1] != scaler.n_features_in_:
            st.error(f"❌ ERREUR FATALE: Le scaler attend {scaler.n_features_in_} features, mais reçoit {features_df.shape[1]}")
            st.write("Colonnes actuelles:", list(features_df.columns))
            st.stop()
        
        # Conversion en array numpy
        features_array = features_df.values.astype(np.float64)
        
        if debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.markdown("### 🐛 DEBUG - Étape 3: Features après encodage")
            st.dataframe(features_df.T, use_container_width=True)
            st.markdown(f"**Shape:** {features_array.shape} | Features: {list(features_df.columns)}")
            st.markdown(f"**Min/Max values:** {features_array.min():.2f} / {features_array.max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Standardisation
        features_scaled = scaler.transform(features_array)
        
        if debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.markdown("### 🐛 DEBUG - Étape 4: Features après standardisation")
            st.dataframe(pd.DataFrame(features_scaled, columns=COLUMNS_ORDER).T, use_container_width=True)
            st.markdown(f"**Min/Max scaled:** {features_scaled.min():.2f} / {features_scaled.max():.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Prédiction
        cluster = kmeans.predict(features_scaled)[0]
        
        # Calculer les distances aux centres
        distances = kmeans.transform(features_scaled)[0]
        confidence = 1 - (distances[cluster] / distances.sum())
        
        if debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.markdown("### 🐛 DEBUG - Étape 5: Prédiction finale")
            st.markdown(f"**Cluster prédit:** {cluster}")
            st.markdown(f"**Distance au centre du cluster {cluster}:** {distances[cluster]:.4f}")
            st.markdown("**Distances à tous les centres:**")
            for i, dist in enumerate(distances):
                st.text(f"  Cluster {i}: {dist:.4f} {'← PRÉDIT' if i == cluster else ''}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ========== AFFICHAGE ENRICHI ==========
        
        st.markdown("---")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown("<h2 style='color: #0d4a6b; margin-bottom: 10px;'>✅ Résultat de la prédiction</h2>", unsafe_allow_html=True)
        st.markdown(f'<div class="cluster-badge">Cluster {cluster}</div>', unsafe_allow_html=True)
        
        if cluster in CLUSTER_PROFILES:
            profile = CLUSTER_PROFILES[cluster]
            st.markdown(f"<h3 style='color: {profile['color']};'>{profile['name']}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: 16px;'>{profile['description']}</p>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Niveau de confiance
        st.markdown("### 📊 Niveau de confiance")
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%;"></div>
        </div>
        <p style='text-align: center;'>Confiance: <strong>{confidence*100:.1f}%</strong></p>
        """, unsafe_allow_html=True)
        
        # Profil détaillé du cluster
        if cluster in CLUSTER_PROFILES:
            profile = CLUSTER_PROFILES[cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔬 Caractéristiques du cluster")
                for char in profile['characteristics']:
                    st.markdown(f"- {char}")
                
                st.markdown(f"### 🎯 Rôle biologique")
                st.info(profile['biological_role'])
            
            with col2:
                st.markdown("### 📚 Exemples de gènes connus")
                for example in profile['examples']:
                    st.markdown(f"- **{example}**")
                
                st.markdown("### 💡 Utilité")
                st.success(f"""
                En classant votre gène dans le **Cluster {cluster}**, vous pouvez:
                - Explorer des gènes similaires bien caractérisés
                - Formuler des hypothèses sur sa fonction
                - Identifier des collaborations possibles
                """)
        
        # Comparaison avec autres clusters
        with st.expander("📊 Comparaison avec les autres clusters"):
            st.markdown("**Distance par rapport aux centres des clusters:**")
            
            distance_df = pd.DataFrame({
                'Cluster': [f"Cluster {i}" for i in range(len(distances))],
                'Distance': distances,
                'Similarité (%)': [100 * (1 - d/distances.sum()) for d in distances]
            }).sort_values('Distance')
            
            st.dataframe(distance_df, use_container_width=True)
            
            st.markdown(f"""
            ✅ Votre gène est **le plus proche du Cluster {cluster}**  
            ℹ️ Plus la distance est faible, plus la similarité est forte
            """)
        
        # Détails techniques
        with st.expander("🔧 Détails techniques"):
            st.write("**Vecteur de features final (19 dimensions):**")
            st.dataframe(features_df.T, use_container_width=True)
            st.write(f"**Modèle:** KMeans avec {kmeans.n_clusters} clusters")
            st.write(f"**Silhouette Score:** 0.326")
            st.write(f"**Centres de clusters:** {kmeans.cluster_centers_.shape}")
    
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>🧬 Application de clustering génomique • 70 620 gènes humains • NCBI Gene Database</p>
    <p style='font-size: 12px; color: #999;'>Clustering basé sur caractéristiques structurelles et fonctionnelles</p>
</div>
""", unsafe_allow_html=True)