# =====================================================
# File: vikor_streamlit_navigation.py
# Sistem Pendukung Keputusan Metode VIKOR (Streamlit)
# VERSI DASHBOARD DENGAN NAVIGASI
# =====================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas

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
# FUNGSI GENERATE PDF
# ---------------------------------------
def generate_pdf(result_df, f_star, f_minus, criteria, criterion_types, weights, alternatives):
    """Generate PDF report dari hasil perhitungan VIKOR"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    
    # Container untuk elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("HASIL PERHITUNGAN METODE VIKOR", title_style)
    elements.append(title)
    
    subtitle = Paragraph(
        f"Sistem Pendukung Keputusan Rekomendasi Laptop<br/>Tanggal: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", 
        ParagraphStyle('subtitle', parent=normal_style, alignment=TA_CENTER, fontSize=10)
    )
    elements.append(subtitle)
    elements.append(Spacer(1, 20))
    
    # Section 1: Ringkasan Perhitungan
    elements.append(Paragraph("1. RINGKASAN PERHITUNGAN", heading_style))
    summary_data = [
        ['Jumlah Alternatif', str(len(alternatives))],
        ['Jumlah Kriteria', str(len(criteria))],
        ['Metode', 'VIKOR'],
        ['Parameter v', '0.5']
    ]
    
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f7fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#667eea')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Section 2: Kriteria dan Bobot
    elements.append(Paragraph("2. KRITERIA DAN BOBOT", heading_style))
    criteria_data = [['No', 'Nama Kriteria', 'Tipe', 'Bobot']]
    for i, (crit, ctype, weight) in enumerate(zip(criteria, criterion_types, weights)):
        criteria_data.append([
            str(i+1),
            crit,
            'Benefit' if ctype == 'benefit' else 'Cost',
            f"{weight:.4f}"
        ])
    
    criteria_table = Table(criteria_data, colWidths=[0.5*inch, 2.5*inch, 1.5*inch, 1.5*inch])
    criteria_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#667eea')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
    ]))
    elements.append(criteria_table)
    elements.append(Spacer(1, 20))
    
    # Section 3: Nilai Ideal dan Anti-Ideal
    elements.append(Paragraph("3. NILAI IDEAL (f*) DAN ANTI-IDEAL (f-)", heading_style))
    ideal_data = [['Kriteria', 'f* (Ideal)', 'f- (Anti-Ideal)']]
    for i, crit in enumerate(criteria):
        ideal_data.append([
            crit,
            f"{f_star[i]:.4f}",
            f"{f_minus[i]:.4f}"
        ])
    
    ideal_table = Table(ideal_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    ideal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#667eea')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
    ]))
    elements.append(ideal_table)
    elements.append(Spacer(1, 20))
    
    # Section 4: HASIL PERANKINGAN (Main Result)
    elements.append(Paragraph("4. HASIL PERANKINGAN VIKOR", heading_style))
    
    # Sort by Rank untuk memastikan urutan 1-9
    result_sorted = result_df.sort_values('Rank').reset_index(drop=True)
    
    ranking_data = [['Rank', 'Alternatif', 'Nilai S', 'Nilai R', 'Nilai Q']]
    for idx, row in result_sorted.iterrows():
        ranking_data.append([
            str(int(row['Rank'])),
            row['Alternative'],
            f"{row['S']:.4f}",
            f"{row['R']:.4f}",
            f"{row['Q']:.4f}"
        ])
    
    ranking_table = Table(ranking_data, colWidths=[0.7*inch, 2.3*inch, 1*inch, 1*inch, 1*inch])
    ranking_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#10b981')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#10b981')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7fa')]),
        # Highlight ranking 1 (baris index 1)
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#d1fae5')),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
    ]))
    elements.append(ranking_table)
    elements.append(Spacer(1, 20))
    
    # Section 5: Rekomendasi
    best_alternative = result_sorted.iloc[0]
    elements.append(Paragraph("5. REKOMENDASI TERBAIK", heading_style))
    
    recommendation_text = f"""
    <b>Alternatif Terbaik: {best_alternative['Alternative']}</b><br/>
    <br/>
    Nilai Q: {best_alternative['Q']:.4f}<br/>
    Nilai S: {best_alternative['S']:.4f}<br/>
    Nilai R: {best_alternative['R']:.4f}<br/>
    <br/>
    Alternatif ini merupakan pilihan terbaik karena memiliki nilai Q terkecil, 
    yang menunjukkan solusi kompromi terbaik antara kedekatan dengan solusi ideal 
    dan deviasi maksimum yang minimal.
    """
    
    recommendation_para = Paragraph(recommendation_text, normal_style)
    elements.append(recommendation_para)
    elements.append(Spacer(1, 20))
    
    # Section 6: Interpretasi
    elements.append(Paragraph("6. INTERPRETASI HASIL", heading_style))
    interpretation_text = """
    <b>Cara Membaca Hasil:</b><br/>
    <br/>
    • <b>Nilai S</b>: Ukuran kedekatan total ke solusi ideal. Semakin kecil nilai S, 
    semakin baik alternatif tersebut secara keseluruhan.<br/>
    <br/>
    • <b>Nilai R</b>: Ukuran deviasi maksimum dari solusi ideal. Semakin kecil nilai R, 
    semakin konsisten alternatif tersebut di semua kriteria.<br/>
    <br/>
    • <b>Nilai Q</b>: Indeks kompromi yang menggabungkan S dan R dengan parameter v=0.5. 
    Nilai Q terkecil menunjukkan alternatif terbaik.<br/>
    <br/>
    • <b>Rank</b>: Peringkat alternatif berdasarkan nilai Q (1 = terbaik).<br/>
    """
    
    interpretation_para = Paragraph(interpretation_text, normal_style)
    elements.append(interpretation_para)
    
    # Footer
    elements.append(Spacer(1, 30))
    footer_text = Paragraph(
        "Dibuat oleh Kelompok 1 | Metode VIKOR | Sistem Pendukung Keputusan",
        ParagraphStyle('footer', parent=normal_style, alignment=TA_CENTER, fontSize=8, textColor=colors.grey)
    )
    elements.append(footer_text)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------------------------------
# INISIALISASI SESSION STATE
# ---------------------------------------
def init_session_state():
    """Inisialisasi semua session state"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Beranda'
    
    if 'm' not in st.session_state:
        st.session_state.m = 9
    
    if 'n' not in st.session_state:
        st.session_state.n = 6
    
    if 'alternatives' not in st.session_state:
        st.session_state.alternatives = []
    
    if 'criteria' not in st.session_state:
        st.session_state.criteria = []
    
    if 'weights' not in st.session_state:
        st.session_state.weights = []
    
    if 'criterion_types' not in st.session_state:
        st.session_state.criterion_types = []
    
    if 'matrix' not in st.session_state:
        st.session_state.matrix = None
    
    if 'result' not in st.session_state:
        st.session_state.result = None

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
        
        .nav-container {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .nav-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
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
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN BERANDA
# ---------------------------------------
def page_beranda():
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
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📋 Tentang Sistem
    
    Dashboard SPK VIKOR ini adalah sistem pendukung keputusan untuk membantu mahasiswa informatika 
    dalam memilih laptop yang paling sesuai dengan kebutuhan mereka. Sistem ini menggunakan metode 
    VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) yang merupakan metode pengambilan 
    keputusan multi-kriteria.
    
    **Cara Menggunakan:**
    1. Mulai dari menu **Pengaturan Dasar** untuk mengatur jumlah alternatif dan kriteria
    2. Isi **Nama Alternatif** dengan laptop yang akan dibandingkan
    3. Tentukan **Kriteria dan Bobot** sesuai prioritas Anda
    4. Input nilai pada **Matriks Keputusan**
    5. Lihat **Hasil Perhitungan** untuk mendapatkan rekomendasi terbaik
    
    Gunakan menu navigasi di sidebar untuk berpindah antar halaman.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN PENGATURAN DASAR
# ---------------------------------------
def page_pengaturan_dasar():
    st.markdown('<div class="custom-card card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">⚙️ Pengaturan Dasar</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        m = st.number_input("🔢 Jumlah Alternatif", min_value=2, max_value=20, value=st.session_state.m, step=1)
    with col2:
        n = st.number_input("📊 Jumlah Kriteria", min_value=2, max_value=10, value=st.session_state.n, step=1)
    
    if st.button("💾 Simpan Pengaturan", use_container_width=True):
        st.session_state.m = m
        st.session_state.n = n
        st.success(f"✅ Pengaturan berhasil disimpan: {m} Alternatif, {n} Kriteria")
        st.info("📍 Lanjutkan ke halaman **Nama Alternatif** di menu navigasi")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN NAMA ALTERNATIF
# ---------------------------------------
def page_nama_alternatif():
    history = load_history()
    
    st.markdown('<div class="custom-card card-green">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💻 Nama Alternatif</div>', unsafe_allow_html=True)
    st.info("💡 Ketik nama laptop, sistem akan mengingat input sebelumnya")
    
    col_alt = st.columns(3)
    alternatives = []
    
    for i in range(int(st.session_state.m)):
        with col_alt[i % 3]:
            suggestion_text = ""
            if history['alternatives']:
                suggestion_text = f"Saran: {', '.join(history['alternatives'][:3])}"
            
            default_value = st.session_state.alternatives[i] if i < len(st.session_state.alternatives) else f"A{i+1}"
            
            alt_input = st.text_input(
                f"Alternatif {i+1}", 
                value=default_value, 
                key=f"alt_{i}",
                help=suggestion_text if suggestion_text else "Masukkan nama laptop"
            )
            alternatives.append(alt_input)
    
    if st.button("💾 Simpan Alternatif", use_container_width=True):
        st.session_state.alternatives = alternatives
        for alt in alternatives:
            if alt and alt not in [f"A{i+1}" for i in range(20)]:
                save_to_history('alternatives', alt)
        st.success("✅ Nama alternatif berhasil disimpan!")
        st.info("📍 Lanjutkan ke halaman **Kriteria dan Bobot** di menu navigasi")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN KRITERIA DAN BOBOT
# ---------------------------------------
def page_kriteria_bobot():
    history = load_history()
    
    st.markdown('<div class="custom-card card-purple">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Kriteria dan Bobot</div>', unsafe_allow_html=True)
    st.info("💡 Ketik nama kriteria, sistem akan mengingat input sebelumnya")
    
    criteria = []
    weights = []
    criterion_types = []
    
    for j in range(int(st.session_state.n)):
        st.markdown(f"**Kriteria {j+1}**")
        col_crit = st.columns([3, 2, 2])
        
        with col_crit[0]:
            suggestion_text = ""
            if history['criteria']:
                suggestion_text = f"Saran: {', '.join(history['criteria'][:3])}"
            
            default_crit = st.session_state.criteria[j] if j < len(st.session_state.criteria) else f"C{j+1}"
            
            crit_input = st.text_input(
                "Nama Kriteria", 
                value=default_crit, 
                key=f"crit_{j}",
                help=suggestion_text if suggestion_text else "Masukkan nama kriteria",
                label_visibility="collapsed"
            )
            criteria.append(crit_input)
            
        with col_crit[1]:
            default_weight = st.session_state.weights[j] if j < len(st.session_state.weights) else 0.15
            weights.append(st.number_input(
                "Bobot", 
                min_value=0.0, 
                max_value=1.0, 
                step=0.05, 
                value=float(default_weight), 
                key=f"weight_{j}"
            ))
        
        with col_crit[2]:
            default_type = st.session_state.criterion_types[j] if j < len(st.session_state.criterion_types) else "benefit"
            criterion_types.append(st.selectbox(
                "Jenis", 
                ["benefit", "cost"], 
                index=0 if default_type == "benefit" else 1,
                key=f"type_{j}"
            ))
        
        st.markdown("---")
    
    if st.button("💾 Simpan Kriteria dan Bobot", use_container_width=True):
        st.session_state.criteria = criteria
        st.session_state.weights = weights
        st.session_state.criterion_types = criterion_types
        
        for crit in criteria:
            if crit and crit not in [f"C{i+1}" for i in range(20)]:
                save_to_history('criteria', crit)
        
        st.success("✅ Kriteria dan bobot berhasil disimpan!")
        st.info("📍 Lanjutkan ke halaman **Matriks Keputusan** di menu navigasi")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN MATRIKS KEPUTUSAN
# ---------------------------------------
def page_matriks_keputusan():
    if not st.session_state.alternatives or not st.session_state.criteria:
        st.warning("⚠️ Silakan lengkapi **Nama Alternatif** dan **Kriteria** terlebih dahulu!")
        return
    
    st.markdown('<div class="custom-card card-yellow">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Matriks Keputusan</div>', unsafe_allow_html=True)
    st.info("📝 Masukkan nilai untuk setiap alternatif dan kriteria")
    
    # Initialize matrix if not exists
    if st.session_state.matrix is None:
        st.session_state.matrix = np.zeros((st.session_state.m, st.session_state.n))
    
    data = []
    for i in range(int(st.session_state.m)):
        st.markdown(f"**{st.session_state.alternatives[i]}**")
        cols = st.columns(int(st.session_state.n))
        row = []
        for j in range(int(st.session_state.n)):
            with cols[j]:
                default_val = st.session_state.matrix[i, j] if st.session_state.matrix is not None else 0.0
                val = st.number_input(
                    f"{st.session_state.criteria[j]}", 
                    step=0.01, 
                    format="%.2f",
                    value=float(default_val),
                    key=f"val_{i}_{j}"
                )
                row.append(val)
        data.append(row)
        st.markdown("---")
    
    matrix = np.array(data)
    
    if st.button("💾 Simpan Matriks", use_container_width=True):
        st.session_state.matrix = matrix
        st.success("✅ Matriks keputusan berhasil disimpan!")
        st.info("📍 Lanjutkan ke halaman **Hasil Perhitungan** untuk melihat hasil analisis")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------
# HALAMAN HASIL PERHITUNGAN
# ---------------------------------------
def page_hasil_perhitungan():
    if st.session_state.matrix is None:
        st.warning("⚠️ Silakan lengkapi **Matriks Keputusan** terlebih dahulu!")
        return
    
    if st.button("🚀 Hitung Metode VIKOR", use_container_width=True):
        weights_array = np.array(st.session_state.weights)
        if np.sum(weights_array) > 0:
            weights_array = weights_array / np.sum(weights_array)
        
        result, f_star, f_minus = vikor(
            st.session_state.matrix, 
            weights_array, 
            st.session_state.criterion_types, 
            v=0.5
        )
        
        result["Alternative"] = [st.session_state.alternatives[int(alt[1:])-1] for alt in result["Alternative"]]
        st.session_state.result = result
        st.session_state.f_star = f_star
        st.session_state.f_minus = f_minus
        
        st.success("✅ Perhitungan selesai!")
    
    if st.session_state.result is not None:
        result = st.session_state.result
        f_star = st.session_state.f_star
        f_minus = st.session_state.f_minus
        
        # NILAI IDEAL
        st.markdown('<div class="custom-card card-indigo">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">⭐ Nilai Ideal (f*) dan Anti-Ideal (f-)</div>', unsafe_allow_html=True)
        
        ideal_df = pd.DataFrame({
            'Kriteria': st.session_state.criteria,
            'Tipe': st.session_state.criterion_types,
            'f* (Ideal)': [f"{val:.4f}" for val in f_star],
            'f- (Anti-Ideal)': [f"{val:.4f}" for val in f_minus],
            'Bobot': [f"{val:.4f}" for val in st.session_state.weights]
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
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            st.download_button(
                label="📥 Download Hasil CSV",
                data=csv,
                file_name=f"hasil_vikor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_download2:
            pdf_buffer = generate_pdf(
                result, 
                f_star, 
                f_minus, 
                st.session_state.criteria,
                st.session_state.criterion_types,
                st.session_state.weights,
                st.session_state.alternatives
            )
            
            st.download_button(
                label="📄 Download Laporan PDF",
                data=pdf_buffer,
                file_name=f"laporan_vikor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
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

# ---------------------------------------
# MAIN APP
# ---------------------------------------
st.set_page_config(
    page_title="Dashboard SPK VIKOR",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi
init_session_state()
load_custom_css()

# HEADER
st.markdown("""
<div class="main-header">
    <h1>💻 Dashboard SPK VIKOR</h1>
    <p>Sistem Pendukung Keputusan Rekomendasi Laptop Terbaik untuk Mahasiswa Informatika</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown('<div class="nav-title">🧭 Menu Navigasi</div>', unsafe_allow_html=True)
    
    pages = {
        "🏠 Beranda": "Beranda",
        "⚙️ Pengaturan Dasar": "Pengaturan Dasar",
        "💻 Nama Alternatif": "Nama Alternatif",
        "📋 Kriteria dan Bobot": "Kriteria dan Bobot",
        "📊 Matriks Keputusan": "Matriks Keputusan",
        "🏆 Hasil Perhitungan": "Hasil Perhitungan"
    }
    
    for label, page_name in pages.items():
        if st.button(label, key=f"nav_{page_name}", use_container_width=True):
            st.session_state.current_page = page_name
    
    st.markdown("---")
    st.markdown(f"**Halaman Aktif:**  \n`{st.session_state.current_page}`")

# ROUTING HALAMAN
if st.session_state.current_page == "Beranda":
    page_beranda()
elif st.session_state.current_page == "Pengaturan Dasar":
    page_pengaturan_dasar()
elif st.session_state.current_page == "Nama Alternatif":
    page_nama_alternatif()
elif st.session_state.current_page == "Kriteria dan Bobot":
    page_kriteria_bobot()
elif st.session_state.current_page == "Matriks Keputusan":
    page_matriks_keputusan()
elif st.session_state.current_page == "Hasil Perhitungan":
    page_hasil_perhitungan()

# FOOTER
st.markdown("""
<div class="footer">
    <h3 style="margin: 0;">Dibuat oleh Kelompok 1</h3>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Metode VIKOR | Sistem Pendukung Keputusan</p>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.7; font-size: 0.9rem;">Dashboard Professional Edition with Navigation</p>
</div>
""", unsafe_allow_html=True)