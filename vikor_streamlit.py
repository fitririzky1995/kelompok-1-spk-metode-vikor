# =====================================================
# File: vikor_streamlit.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# VERSI DASHBOARD PROFESIONAL
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------
# FUNGSI VIKOR (VERSI DIPERBAIKI)
# ---------------------------------------
def vikor(decision_matrix, weights, criterion_types, v=0.5):
    """
    Implementasi Metode VIKOR yang benar
    
    Parameters:
    - decision_matrix: matriks keputusan (m x n)
    - weights: bobot kriteria (array of n)
    - criterion_types: tipe kriteria ['benefit' atau 'cost']
    - v: bobot strategi (default 0.5)
    
    Returns:
    - DataFrame dengan kolom: Alternative, S, R, Q, Rank
    """
    m, n = decision_matrix.shape

    # Step 1: Hitung nilai terbaik (f*) dan terburuk (f-)
    f_star = np.zeros(n)
    f_minus = np.zeros(n)

    for j in range(n):
        if criterion_types[j] == 'benefit':
            f_star[j] = np.max(decision_matrix[:, j])
            f_minus[j] = np.min(decision_matrix[:, j])
        else:  # cost
            f_star[j] = np.min(decision_matrix[:, j])
            f_minus[j] = np.max(decision_matrix[:, j])

    # Step 2: Hitung S(i) & R(i)
    S = np.zeros(m)
    R = np.zeros(m)

    for i in range(m):
        diff_list = []

        for j in range(n):
            # Hitung denominator (range)
            denom = abs(f_star[j] - f_minus[j])
            
            if denom < 1e-9:
                # Jika semua nilai sama, kontribusinya 0
                normalized = 0
            else:
                # Hitung jarak dari solusi ideal
                if criterion_types[j] == 'benefit':
                    distance = (f_star[j] - decision_matrix[i, j]) / denom
                else:  # cost
                    distance = (decision_matrix[i, j] - f_star[j]) / denom
                
                normalized = distance

            diff_list.append(weights[j] * normalized)

        S[i] = np.sum(diff_list)
        R[i] = np.max(diff_list)

    # Step 3: Hitung Q(i)
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    Q = np.zeros(m)
    for i in range(m):
        s_term = 0 if abs(S_minus - S_star) < 1e-9 else (S[i] - S_star) / (S_minus - S_star)
        r_term = 0 if abs(R_minus - R_star) < 1e-9 else (R[i] - R_star) / (R_minus - R_star)
        
        Q[i] = v * s_term + (1 - v) * r_term

    # Step 4: Tabel hasil
    df = pd.DataFrame({
        'Alternative': [f"A{i+1}" for i in range(m)],
        'S': S,
        'R': R,
        'Q': Q
    })

    df['Rank'] = df['Q'].rank(method='min').astype(int)
    df = df.sort_values(by='Q').reset_index(drop=True)

    return df, f_star, f_minus

# ---------------------------------------
# FUNGSI UNTUK MENYIMPAN & LOAD HISTORY
# ---------------------------------------
def load_history():
    """Load history dari session state"""
    if 'history' not in st.session_state:
        st.session_state.history = {
            'alternatives': [],
            'criteria': []
        }
    return st.session_state.history

def save_to_history(key, value):
    """Simpan nilai ke history"""
    history = load_history()
    if value and value not in history[key]:
        history[key].append(value)
        st.session_state.history = history

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
def load_custom_css():
    st.markdown("""
    <style>
        @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css');
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin: 0.5rem 0 0 0;
            opacity: 0.95;
        }
        
        .team-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            border-left: 5px solid #667eea;
        }
        
        .team-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .team-member {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            transition: transform 0.3s ease;
        }
        
        .team-member:hover {
            transform: translateX(5px);
        }
        
        .member-name {
            font-weight: 700;
            font-size: 1.1rem;
            color: #2d3748;
            margin-bottom: 0.3rem;
        }
        
        .member-role {
            font-size: 0.9rem;
            color: #667eea;
            font-weight: 500;
        }
        
        .custom-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-top: 4px solid;
        }
        
        .card-blue { border-top-color: #3b82f6; }
        .card-green { border-top-color: #10b981; }
        .card-purple { border-top-color: #8b5cf6; }
        .card-yellow { border-top-color: #f59e0b; }
        .card-red { border-top-color: #ef4444; }
        .card-indigo { border-top-color: #6366f1; }
        
        .card-title {
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 700;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            border: none;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
            font-size: 1.1rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            border-radius: 8px;
            border: 2px solid #e2e8f0;
            padding: 0.5rem;
            transition: border-color 0.3s ease;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        .success-box {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #10b981;
            margin: 1rem 0;
        }
        
        .success-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #065f46;
            margin-bottom: 0.5rem;
        }
        
        .success-value {
            font-size: 2rem;
            font-weight: 800;
            color: #10b981;
        }
        
        .info-box {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.15) 100%);
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #3b82f6;
            margin: 1rem 0;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        
        .badge-benefit {
            background-color: rgba(16, 185, 129, 0.25);
            color: #065f46;
        }
        
        .badge-cost {
            background-color: rgba(239, 68, 68, 0.25);
            color: #991b1b;
        }
        
        .footer {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 3rem;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(
    page_title="Dashboard SPK VIKOR",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_custom_css()
history = load_history()

# HEADER
st.markdown("""
<div class="main-header">
    <h1>💻 Dashboard SPK VIKOR</h1>
    <p>Sistem Pendukung Keputusan Rekomendasi Laptop Terbaik untuk Mahasiswa Informatika</p>
</div>
""", unsafe_allow_html=True)

# TEAM MEMBERS
st.markdown("""
<div class="team-section">
    <div class="team-title">
        👥 Tim Pengembang
    </div>
    <div class="team-grid">
        <div class="team-member">
            <div class="member-name">M Ziran</div>
            <div class="member-role">Data Engineer / Data Analyst</div>
        </div>
        <div class="team-member">
            <div class="member-name">Syerly</div>
            <div class="member-role">Research Analyst / Literature Reviewer</div>
        </div>
        <div class="team-member">
            <div class="member-name">Hernan</div>
            <div class="member-role">Full Stack Data Application Developer</div>
        </div>
        <div class="team-member">
            <div class="member-name">Rizky</div>
            <div class="member-role">Technical Writer / Documentation Engineer</div>
        </div>
        <div class="team-member">
            <div class="member-name">Farhan</div>
            <div class="member-role">Data Validation & Computation Analyst</div>
        </div>
        <div class="team-member">
            <div class="member-name">Sarifudin</div>
            <div class="member-role">Quality Assurance (QA) / Project Controller</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# INPUT DASAR
st.markdown('<div class="card-title">⚙️ Pengaturan Dasar</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    m = st.number_input("🔢 Jumlah Alternatif", min_value=2, max_value=20, value=9, step=1)
with col2:
    n = st.number_input("📊 Jumlah Kriteria", min_value=2, max_value=10, value=6, step=1)

if m and n:
    with st.form("input_form"):
        # NAMA ALTERNATIF
        st.markdown('<div class="custom-card card-green">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💻 Nama Alternatif</div>', unsafe_allow_html=True)
        st.info("💡 Ketik nama laptop, sistem akan mengingat input sebelumnya")
        
        col_alt = st.columns(3)
        alternatives = []
        for i in range(int(m)):
            with col_alt[i % 3]:
                suggestion_text = ""
                if history['alternatives']:
                    suggestion_text = f"Saran: {', '.join(history['alternatives'][:3])}"
                
                alt_input = st.text_input(
                    f"Alternatif {i+1}", 
                    value=f"A{i+1}", 
                    key=f"alt_{i}",
                    help=suggestion_text if suggestion_text else "Masukkan nama laptop"
                )
                alternatives.append(alt_input)
                
        st.markdown('</div>', unsafe_allow_html=True)

        # KRITERIA DAN BOBOT
        st.markdown('<div class="custom-card card-purple">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Kriteria dan Bobot</div>', unsafe_allow_html=True)
        st.info("💡 Ketik nama kriteria, sistem akan mengingat input sebelumnya")
        
        criteria = []
        weights = []
        criterion_types = []
        
        for j in range(int(n)):
            st.markdown(f"**Kriteria {j+1}**")
            col_crit = st.columns([3, 2, 2])
            
            with col_crit[0]:
                suggestion_text = ""
                if history['criteria']:
                    suggestion_text = f"Saran: {', '.join(history['criteria'][:3])}"
                
                crit_input = st.text_input(
                    "Nama Kriteria", 
                    value=f"C{j+1}", 
                    key=f"crit_{j}",
                    help=suggestion_text if suggestion_text else "Masukkan nama kriteria",
                    label_visibility="collapsed"
                )
                criteria.append(crit_input)
                
            with col_crit[1]:
                weights.append(st.number_input(
                    "Bobot", 
                    min_value=0.0, 
                    max_value=1.0, 
                    step=0.05, 
                    value=0.15, 
                    key=f"weight_{j}"
                ))
            with col_crit[2]:
                criterion_types.append(st.selectbox(
                    "Jenis", 
                    ["benefit", "cost"], 
                    key=f"type_{j}"
                ))
            
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)

        # MATRIKS KEPUTUSAN
        st.markdown('<div class="custom-card card-yellow">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Matriks Keputusan</div>', unsafe_allow_html=True)
        st.info("📝 Masukkan nilai untuk setiap alternatif dan kriteria")
        
        data = []
        for i in range(int(m)):
            st.markdown(f"**{alternatives[i] if alternatives[i] else f'A{i+1}'}**")
            cols = st.columns(int(n))
            row = []
            for j in range(int(n)):
                with cols[j]:
                    val = st.number_input(
                        f"{criteria[j] if criteria[j] else f'C{j+1}'}", 
                        step=0.01, 
                        format="%.2f",
                        key=f"val_{i}_{j}"
                    )
                    row.append(val)
            data.append(row)
            st.markdown("---")
        
        matrix = np.array(data)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Hitung Metode VIKOR", use_container_width=True)

    # PROSES PERHITUNGAN
    if submit:
        for alt in alternatives:
            if alt and alt not in [f"A{i+1}" for i in range(20)]:
                save_to_history('alternatives', alt)
        
        for crit in criteria:
            if crit and crit not in [f"C{i+1}" for i in range(20)]:
                save_to_history('criteria', crit)
        
        weights_array = np.array(weights)
        if np.sum(weights_array) > 0:
            weights_array = weights_array / np.sum(weights_array)
        
        result, f_star, f_minus = vikor(matrix, weights_array, criterion_types, v=0.5)
        result["Alternative"] = [alternatives[int(alt[1:])-1] for alt in result["Alternative"]]

        st.success("✅ Perhitungan selesai!")
        
        # NILAI IDEAL
        st.markdown('<div class="custom-card card-indigo">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⭐ Nilai Ideal (f*) dan Anti-Ideal (f-)</div>', unsafe_allow_html=True)
        
        ideal_df = pd.DataFrame({
            'Kriteria': criteria,
            'Tipe': criterion_types,
            'f* (Ideal)': [f"{val:.4f}" for val in f_star],
            'f- (Anti-Ideal)': [f"{val:.4f}" for val in f_minus],
            'Bobot': [f"{val:.4f}" for val in weights_array]
        })
        
        st.dataframe(ideal_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # HASIL PERANKINGAN
        st.markdown('<div class="custom-card card-green">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🏆 Hasil Perankingan VIKOR</div>', unsafe_allow_html=True)
        
        display_result = result.copy()
        display_result['S'] = display_result['S'].round(4)
        display_result['R'] = display_result['R'].round(4)
        display_result['Q'] = display_result['Q'].round(4)
        
        st.dataframe(display_result, use_container_width=True, hide_index=True)

        # REKOMENDASI TERBAIK
        best = result.iloc[0]
        st.markdown(f"""
        <div class="success-box">
            <div class="success-title">🏆 Rekomendasi Terbaik</div>
            <div class="success-value">{best['Alternative']}</div>
            <p style="margin-top: 1rem; color: #065f46; font-size: 1.1rem;">
                <strong>Nilai Q:</strong> {best['Q']:.4f} | 
                <strong>Nilai S:</strong> {best['S']:.4f} | 
                <strong>Nilai R:</strong> {best['R']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # VISUALISASI
        st.markdown('<div class="custom-card card-blue">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Visualisasi Nilai Q</div>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=result['Alternative'],
                y=result['Q'],
                marker=dict(
                    color=result['Q'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Nilai Q")
                ),
                text=result['Q'].round(4),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Perbandingan Nilai Q (Semakin Kecil Semakin Baik)",
            xaxis_title="Alternatif",
            yaxis_title="Nilai Q",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # DOWNLOAD
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Hasil CSV",
            data=csv,
            file_name="hasil_vikor.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # INTERPRETASI
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ℹ️ Interpretasi Hasil</div>', unsafe_allow_html=True)
        st.markdown("""
        **Cara Membaca Hasil:**
        - **Nilai S**: Ukuran kedekatan total ke solusi ideal (semakin kecil semakin baik)
        - **Nilai R**: Ukuran deviasi maksimum dari solusi ideal (semakin kecil semakin baik)
        - **Nilai Q**: Indeks kompromi yang menggabungkan S dan R (semakin kecil semakin baik)
        - **Rank**: Peringkat alternatif (1 = terbaik)
        
        Alternatif dengan nilai Q terkecil adalah pilihan terbaik karena:
        1. Paling dekat dengan solusi ideal secara keseluruhan (S kecil)
        2. Memiliki deviasi maksimum yang kecil (R kecil)
        3. Memberikan solusi kompromi yang seimbang
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <h3 style="margin: 0;">Dibuat oleh Kelompok 1</h3>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Metode VIKOR | Sistem Pendukung Keputusan</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.9rem;">Dashboard Professional Edition</p>
</div>
""", unsafe_allow_html=True)