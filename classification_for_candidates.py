import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------
# 1. CẤU HÌNH GIAO DIỆN CHÍNH
# ---------------------------------------------------------
st.set_page_config(
    page_title="ITViec Analytics & AI Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. BỘ CSS ĐA GIAO DIỆN (TỰ ĐỘNG THÍCH ỨNG LIGHT / DARK MODE)
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 60%, #0284c7 100%);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 32px 36px;
    color: #ffffff !important;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.3);
}
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff !important;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 15px;
    color: #cbd5e1 !important;
    line-height: 1.6;
    max-width: 850px;
}

/* Thẻ tính năng (Feature Box) */
.feature-box {
    background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.05));
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 22px;
    height: 100%;
    transition: all 0.2s ease-in-out;
}
.feature-box:hover {
    transform: translateY(-3px);
    border-color: #38bdf8;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.15);
}
.feature-icon {
    font-size: 26px;
    margin-bottom: 10px;
    display: inline-block;
}
.feature-title {
    margin: 0 0 8px 0;
    font-weight: 700;
    font-size: 17px;
    color: var(--text-color, #f8fafc) !important;
}
.feature-desc {
    font-size: 13.5px;
    color: var(--text-color, #cbd5e1) !important;
    opacity: 0.9;
    line-height: 1.6;
}

/* Thẻ thành viên nhóm */
.team-card-modern {
    background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.05));
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.2s ease;
}
.team-card-modern:hover {
    transform: translateY(-3px);
    border-color: #f59e0b;
}
.avatar-circle {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: linear-gradient(135deg, #0284c7, #0f172a);
    color: #ffffff !important;
    font-weight: 700;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 12px auto;
    border: 2px solid #38bdf8;
}
.team-name {
    font-weight: 700;
    font-size: 17px;
    color: var(--text-color, #f8fafc) !important;
    margin-bottom: 4px;
}
.team-role {
    font-size: 13px;
    color: var(--text-color, #94a3b8) !important;
    opacity: 0.85;
    margin-bottom: 12px;
    min-height: 38px;
}

/* Thẻ kết quả công ty */
.company-card {
    background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.05));
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
    transition: all 0.2s ease;
}
.company-card:hover {
    border-color: #38bdf8;
    box-shadow: 0 6px 14px rgba(2, 132, 199, 0.2);
}
.company-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.company-name-title {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-color, #f8fafc) !important;
}

/* Huy hiệu (Badge) tương phản cao */
.badge-pill {
    padding: 5px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
}
.badge-blue { background-color: #0284c7; color: #ffffff !important; }
.badge-green { background-color: #16a34a; color: #ffffff !important; }
.badge-amber { background-color: #d97706; color: #ffffff !important; }

/* Nút bấm bo góc */
div.stButton > button {
    border-radius: 12px;
    font-weight: 600;
    padding: 8px 22px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. TIỀN XỬ LÝ DỮ LIỆU
# ---------------------------------------------------------
DEFAULT_STOP_WORDS = set(ENGLISH_STOP_WORDS).union({
    "a", "an", "the", "in", "on", "at", "to", "from", "by", "of", "with",
    "and", "but", "or", "for", "nor", "so", "yet", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "be", "have", "do",
    "https", "www", "com", "vn"
})

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(
        r"[^a-zA-Z\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
        " ",
        text,
    )
    words = text.split()
    return " ".join([w for w in words if w not in DEFAULT_STOP_WORDS and len(w) > 1])

@st.cache_data
def load_and_preprocess_data():
    comp_file = "Overview_Companies.xlsx"
    rev_file = "Reviews.xlsx"
    
    if not os.path.exists(comp_file) or not os.path.exists(rev_file):
        st.error("⚠️ Không tìm thấy các file dữ liệu Excel. Vui lòng kiểm tra lại thư mục!")
        st.stop()
        
    df_comp = pd.read_excel(comp_file)
    df_rev = pd.read_excel(rev_file)

    df_comp["profile"] = (
        df_comp["Company overview"].fillna("") + " " +
        df_comp["Our key skills"].fillna("") + " " +
        df_comp["Why you'll love working here"].fillna("") + " " +
        df_comp["Company industry"].fillna("") + " " +
        df_comp["Company Type"].fillna("")
    )
    df_comp["clean_profile"] = df_comp["profile"].apply(clean_text)

    num_cols = [
        "Rating", "Salary & benefits", "Training & learning",
        "Management cares about me", "Culture & fun", "Office & workspace"
    ]
    
    comp_avg_ratings = df_rev.groupby("Company Name")[num_cols].mean().round(2).reset_index()
    comp_review_counts = df_rev.groupby("Company Name")["id"].count().rename("Review_Count").reset_index()
    
    df_rev["rec_num"] = df_rev["Recommend?"].map({"Yes": 1, "No": 0})
    comp_rec_pct = (df_rev.groupby("Company Name")["rec_num"].mean() * 100).round(1).rename("Actual_Recommend_Rate").reset_index()

    df_comp_merged = df_comp.merge(comp_avg_ratings, on="Company Name", how="left")
    df_comp_merged = df_comp_merged.merge(comp_review_counts, on="Company Name", how="left")
    df_comp_merged = df_comp_merged.merge(comp_rec_pct, on="Company Name", how="left")

    return df_comp_merged, df_rev, num_cols

@st.cache_resource
def build_models(clean_profiles, _df_rev, num_features):
    vec = TfidfVectorizer(min_df=1)
    tfidf_mat = vec.fit_transform(clean_profiles)
    cos_sim = cosine_similarity(tfidf_mat, tfidf_mat)

    X = _df_rev[num_features].fillna(_df_rev[num_features].mean())
    y = _df_rev["Recommend?"].map({"Yes": 1, "No": 0}).fillna(1).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_scaled, y)

    return vec, tfidf_mat, cos_sim, scaler, knn

df_companies, df_reviews, num_features = load_and_preprocess_data()
vectorizer, tfidf_matrix, cosine_sim, scaler, classifier_model = build_models(
    df_companies["clean_profile"].tolist(), df_reviews, num_features
)

# ---------------------------------------------------------
# HÀM HIỂN THỊ HỒ SƠ CHI TIẾT CÔNG TY (TƯƠNG THÍCH MỌI GIAO DIỆN)
# ---------------------------------------------------------
def display_company_details(company_name, df_data):
    row = df_data[df_data["Company Name"] == company_name]
    if row.empty:
        st.warning("Không tìm thấy thông tin chi tiết.")
        return
    
    comp = row.iloc[0]
    
    st.markdown(f"""
    <div style="background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.05)); border:1px solid rgba(148, 163, 184, 0.3); border-radius:18px; padding:20px 24px; margin-top:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(148, 163, 184, 0.2); padding-bottom:12px; margin-bottom:14px;">
            <div>
                <h3 style="margin:0; font-size:22px; color:var(--text-color, #f8fafc); font-weight:800;">🏢 {comp['Company Name']}</h3>
                <span style="font-size:14px; opacity:0.85;">📍 {comp.get('Location', 'Việt Nam')}</span>
            </div>
            <div>
                <span class="badge-pill badge-blue">{comp.get('Company industry', 'Công nghệ')}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Thẻ điểm KPI
    if not pd.isna(comp.get("Rating")):
        st.markdown("#### 🌟 Thống kê đánh giá nhân sự thực tế")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("⭐ Rating Tổng Quan", f"{comp['Rating']} / 5.0")
            st.metric("💰 Lương & Phúc lợi", f"{comp['Salary & benefits']} / 5.0")
        with m2:
            st.metric("📚 Đào tạo & Phát triển", f"{comp['Training & learning']} / 5.0")
            st.metric("🤝 Quản lý quan tâm", f"{comp['Management cares about me']} / 5.0")
        with m3:
            st.metric("🎉 Văn hoá & Con người", f"{comp['Culture & fun']} / 5.0")
            st.metric("🏢 Văn phòng làm việc", f"{comp['Office & workspace']} / 5.0")
        with m4:
            rec_rate = f"{comp['Actual_Recommend_Rate']}%" if not pd.isna(comp.get("Actual_Recommend_Rate")) else "N/A"
            rev_cnt = f"{int(comp['Review_Count'])} lượt" if not pd.isna(comp.get("Review_Count")) else "0"
            st.metric("👍 Tỷ lệ Recommend", rec_rate)
            st.metric("📝 Tổng lượt Reviews", rev_cnt)
    else:
        st.info("ℹ️ Công ty này chưa có lượt đánh giá chi tiết trên hệ thống.")

    # Thông tin hoạt động
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""
        **🏢 Thông tin doanh nghiệp:**
        * **Loại hình:** `{comp.get('Company Type', 'N/A')}`
        * **Quy mô:** `{comp.get('Company size', 'N/A')}`
        * **Quốc gia:** `{comp.get('Country', 'N/A')}`
        """)
    with col_r:
        skills = comp.get('Our key skills', 'Đang cập nhật')
        st.markdown(f"""
        **💻 Kỹ năng công nghệ trọng tâm:**  
        `{skills}`
        """)

    with st.expander("📖 Giới thiệu chi tiết về doanh nghiệp", expanded=True):
        st.write(comp.get('Company overview', 'Chưa có thông tin giới thiệu.'))

    why_love = comp.get("Why you'll love working here", '')
    if why_love and str(why_love).strip() != "":
        with st.expander("✨ Vì sao bạn sẽ thích làm việc tại đây?"):
            st.write(why_love)

# ---------------------------------------------------------
# 4. THANH BÊN (SIDEBAR NAVIGATION)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:10px 0 20px 0;">
        <span style="font-size:38px;">⚡</span>
        <h2 style="margin:5px 0 0 0; font-size:22px; font-weight:800; color:var(--text-color, #ffffff);">ITViec AI Hub</h2>
        <span class="badge-pill badge-amber" style="margin-top:6px;">DL07 • K316</span>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "ĐIỀU HƯỚNG CHÍNH",
        [
            "🏠 Giới thiệu & Dự án",
            "🎯 Đề xuất công ty tương tự",
            "👍 Phân loại & Tìm kiếm Recommend"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:12.5px; opacity:0.85; line-height:1.6;">
        <b>Đồ án Khoa học Dữ liệu 2</b><br>
        Ứng dụng Trí tuệ Nhân tạo & Xử lý Ngôn ngữ Tự nhiên trong Khảo sát Tuyển dụng Công nghệ thông tin.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TRANG 1: HERO BANNER & NHIỆM VỤ DỰ ÁN
# =========================================================
if menu == "🏠 Giới thiệu & Dự án":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">ITViec Analytics & AI Intelligence</div>
        <div class="hero-subtitle">
            Nền tảng phân tích dữ liệu tuyển dụng, gợi ý môi trường làm việc tương thích bằng học máy và phân loại mức độ sẵn sàng giới thiệu công ty từ phản hồi của kỹ sư phần mềm.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Các nhiệm vụ nghiên cứu trọng tâm")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="feature-box">
            <span class="feature-icon">🔍</span>
            <div class="feature-title">Nhiệm vụ 1: Content-Based Recommender</div>
            <div class="feature-desc">
                Xây dựng bộ máy đề xuất công ty tương thích bằng <b>TF-IDF</b> và <b>Cosine Similarity</b>. Cho phép ứng viên tìm kiếm môi trường làm việc thông qua hồ sơ công ty hoặc nhập trực tiếp các từ khoá công nghệ (Java, Python, FinTech, Hybrid...).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-box">
            <span class="feature-icon">🤖</span>
            <div class="feature-title">Nhiệm vụ 2: Sentiment Classification & Filtering</div>
            <div class="feature-desc">
                Phân loại mức độ hài lòng của ứng viên (<b>Recommend: Yes/No</b>) qua mô hình Machine Learning được tối ưu trên 6 chỉ số đánh giá. Hỗ trợ lọc các công ty thoả mãn tiêu chuẩn lương thưởng, văn hoá và cập nhật đánh giá theo thời gian thực.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Thống kê nhanh dữ liệu
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📊 Tổng quan cơ sở dữ liệu")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Tổng Doanh Nghiệp IT", f"{len(df_companies)} công ty")
    with s2:
        st.metric("Tổng Lượt Đánh Giá Khảo Sát", f"{len(df_reviews):,} lượt")
    with s3:
        yes_rate = round(df_reviews['Recommend?'].value_counts(normalize=True).get('Yes', 0) * 100, 1)
        st.metric("Tỷ Lệ Đánh Giá Tích Cực", f"{yes_rate}%")

    # Nhóm phát triển ở cuối trang
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h3 style="margin:0; font-weight:800; color:var(--text-color, #ffffff);">👥 Đội ngũ thực hiện dự án</h3>
        <span class="badge-pill badge-amber">Khoá học DL07 • K316</span>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("""
        <div class="team-card-modern">
            <div class="avatar-circle">VT</div>
            <div class="team-name">Nguyễn Văn Tư</div>
            <div class="team-role">PySpark, Anomaly Detection, Product Demo</div>
            <a href="mailto:tunguyen.dg@gmail.com" style="color:#38bdf8; font-size:13px; text-decoration:none; font-weight:600;">tunguyen.dg@gmail.com</a>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="team-card-modern">
            <div class="avatar-circle">KH</div>
            <div class="team-name">Nguyễn Phạm Khánh Hưng</div>
            <div class="team-role">Machine Learning / Regression Modeling</div>
            <a href="mailto:khanhhung1310@gmail.com" style="color:#38bdf8; font-size:13px; text-decoration:none; font-weight:600;">khanhhung1310@gmail.com</a>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown("""
        <div class="team-card-modern">
            <div class="avatar-circle">QR</div>
            <div class="team-name">Nguyễn Thị Mai Quy Ri</div>
            <div class="team-role">EDA, Data Cleaning, Presentation</div>
            <a href="mailto:mari.ng17@icloud.com" style="color:#38bdf8; font-size:13px; text-decoration:none; font-weight:600;">mari.ng17@icloud.com</a>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TRANG 2: NHIỆM VỤ 1 - ĐỀ XUẤT CÔNG TY TƯƠNG ĐỒNG
# =========================================================
elif menu == "🎯 Đề xuất công ty tương tự":
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-size:26px; font-weight:800; color:var(--text-color, #ffffff); margin:0;">🎯 Đề xuất công ty IT tương đồng</h2>
        <span style="font-size:14px; opacity:0.85;">Hệ thống áp dụng thuật toán TF-IDF & Cosine Similarity so khớp dữ liệu ngữ nghĩa.</span>
    </div>
    """, unsafe_allow_html=True)
    
    sub_tab_1 = st.radio(
        "Phương thức đề xuất:",
        ["🏢 Chọn công ty có sẵn để tìm đối thủ tương tự", "🔍 Tìm kiếm thông minh theo từ khoá (Skills/Industry)"],
        horizontal=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- NHÁNH 1: TÌM THEO CÔNG TY CÓ SẴN ---
    if sub_tab_1 == "🏢 Chọn công ty có sẵn để tìm đối thủ tương tự":
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            company_names = df_companies["Company Name"].dropna().tolist()
            selected_company = st.selectbox("Chọn công ty gốc để so sánh:", options=company_names)
        with col_c2:
            top_k_comp = st.slider("Số lượng đề xuất:", 3, 15, 5)

        if st.button("🚀 Phân Tích & Đề Xuất", type="primary"):
            idx = df_companies[df_companies["Company Name"] == selected_company].index[0]
            scores = cosine_sim[idx].copy()
            scores[idx] = -1.0
            
            top_indices = np.argsort(scores)[::-1][:top_k_comp]
            
            rec_comp_list = []
            for rank, c_idx in enumerate(top_indices, 1):
                row = df_companies.iloc[c_idx]
                rec_comp_list.append({
                    "Hạng": f"Top {rank}",
                    "Tên công ty": row["Company Name"],
                    "Độ tương đồng": f"{scores[c_idx]*100:.1f}%",
                    "Score_raw": scores[c_idx],
                    "Lĩnh vực": row["Company industry"],
                    "Quy mô": row["Company size"],
                    "Rating": row.get('Rating', np.nan),
                    "Recommend": row.get('Actual_Recommend_Rate', np.nan)
                })
            st.session_state["rec_by_comp_results"] = rec_comp_list
            st.session_state["rec_by_comp_target"] = selected_company

        if "rec_by_comp_results" in st.session_state and st.session_state.get("rec_by_comp_target") == selected_company:
            st.markdown(f"#### 🏆 Top {len(st.session_state['rec_by_comp_results'])} công ty tương tự với `{selected_company}`:")
            
            for item in st.session_state["rec_by_comp_results"]:
                rating_badge = f"⭐ {item['Rating']}" if not pd.isna(item['Rating']) else "Chưa có sao"
                rec_badge = f"👍 {item['Recommend']}%" if not pd.isna(item['Recommend']) else "Chưa có %"
                
                st.markdown(f"""
                <div class="company-card">
                    <div class="company-card-header">
                        <div>
                            <span class="company-name-title">{item['Tên công ty']}</span>
                            <span class="badge-pill badge-blue" style="margin-left:10px;">{item['Hạng']}</span>
                        </div>
                        <div>
                            <span class="badge-pill badge-green">Độ khớp: {item['Độ tương đồng']}</span>
                        </div>
                    </div>
                    <div style="font-size:13.5px; opacity:0.85; margin-bottom:8px;">
                        🏢 Lĩnh vực: <b>{item['Lĩnh vực']}</b> &nbsp;|&nbsp; 👥 Quy mô: <b>{item['Quy mô']}</b>
                    </div>
                    <div style="display:flex; gap:16px; font-size:13px; font-weight:700;">
                        <span style="color:#f59e0b;">{rating_badge}</span>
                        <span style="color:#10b981;">{rec_badge} khuyên làm việc</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Xem hồ sơ & Thống kê chi tiết")
            rec_names = [item["Tên công ty"] for item in st.session_state["rec_by_comp_results"]]
            selected_detail_comp = st.selectbox("Chọn công ty để xem chi tiết toàn diện:", options=rec_names)
            if selected_detail_comp:
                display_company_details(selected_detail_comp, df_companies)

    # --- NHÁNH 2: TÌM THEO TỪ KHOÁ ---
    else:
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            user_query = st.text_input(
                "Nhập từ khoá kỹ năng, công nghệ hoặc chế độ phúc lợi:",
                placeholder="Ví dụ: Java Spring Boot AWS Fintech Banking hybrid HCM..."
            )
        with col_k2:
            top_k_kw = st.slider("Số lượng đề xuất:", 3, 15, 5, key="top_k_kw")

        if st.button("🔍 Tìm Kiếm Thông Minh", type="primary"):
            if not user_query.strip():
                st.warning("Vui lòng nhập từ khoá tìm kiếm!")
            else:
                clean_query = clean_text(user_query)
                query_vec = vectorizer.transform([clean_query])
                kw_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
                ranked_indices = np.argsort(kw_scores)[::-1][:top_k_kw]
                
                if kw_scores[ranked_indices[0]] == 0.0:
                    st.info("Không tìm thấy công ty nào phù hợp với từ khoá này. Hãy thử từ khoá công nghệ khác!")
                    st.session_state["rec_by_kw_results"] = []
                else:
                    kw_results = []
                    for rank, idx in enumerate(ranked_indices, 1):
                        if kw_scores[idx] > 0:
                            row = df_companies.iloc[idx]
                            kw_results.append({
                                "Hạng": f"Top {rank}",
                                "Tên công ty": row["Company Name"],
                                "Điểm khớp": f"{kw_scores[idx]*100:.1f}%",
                                "Lĩnh vực": row["Company industry"],
                                "Kỹ năng": row.get("Our key skills", "N/A"),
                                "Rating": row.get('Rating', np.nan),
                                "Recommend": row.get('Actual_Recommend_Rate', np.nan)
                            })
                    st.session_state["rec_by_kw_results"] = kw_results
                    st.session_state["rec_by_kw_query"] = user_query

        if "rec_by_kw_results" in st.session_state and len(st.session_state["rec_by_kw_results"]) > 0:
            st.markdown(f"#### 🏆 Kết quả đề xuất phù hợp nhất với từ khoá: *\"{st.session_state.get('rec_by_kw_query', '')}\"*")
            
            for item in st.session_state["rec_by_kw_results"]:
                rating_badge = f"⭐ {item['Rating']}" if not pd.isna(item['Rating']) else "Chưa có sao"
                rec_badge = f"👍 {item['Recommend']}%" if not pd.isna(item['Recommend']) else "Chưa có %"
                
                st.markdown(f"""
                <div class="company-card">
                    <div class="company-card-header">
                        <div>
                            <span class="company-name-title">{item['Tên công ty']}</span>
                            <span class="badge-pill badge-blue" style="margin-left:10px;">{item['Hạng']}</span>
                        </div>
                        <div>
                            <span class="badge-pill badge-green">Độ khớp: {item['Điểm khớp']}</span>
                        </div>
                    </div>
                    <div style="font-size:13.5px; opacity:0.85; margin-bottom:8px;">
                        💻 Kỹ năng: <b>{item['Kỹ năng']}</b>
                    </div>
                    <div style="display:flex; gap:16px; font-size:13px; font-weight:700;">
                        <span style="color:#f59e0b;">{rating_badge}</span>
                        <span style="color:#10b981;">{rec_badge} khuyên làm việc</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Xem hồ sơ & Thống kê chi tiết")
            kw_names = [item["Tên công ty"] for item in st.session_state["rec_by_kw_results"]]
            selected_detail_kw = st.selectbox("Chọn công ty để xem chi tiết toàn diện:", options=kw_names)
            if selected_detail_kw:
                display_company_details(selected_detail_kw, df_companies)

# =========================================================
# TRANG 3: NHIỆM VỤ 2 - PHÂN LOẠI & LỌC THEO ĐÁNH GIÁ
# =========================================================
elif menu == "👍 Phân loại & Tìm kiếm Recommend":
    st.markdown("""
    <div style="margin-bottom:20px;">
        <h2 style="font-size:26px; font-weight:800; color:var(--text-color, #ffffff); margin:0;">👍 Phân loại & Tìm kiếm theo đánh giá (Recommend)</h2>
        <span style="font-size:14px; opacity:0.85;">Hệ thống tự động phân loại mức độ sẵn sàng giới thiệu công ty từ các chỉ số nhân sự.</span>
    </div>
    """, unsafe_allow_html=True)

    if "sub_tab_3" not in st.session_state:
        st.session_state["sub_tab_3"] = "✍️ Tự nhập điểm đánh giá & Cập nhật Rating"

    sub_menu = st.radio(
        "Chế độ tương tác:",
        [
            "✍️ Tự nhập điểm đánh giá & Cập nhật Rating",
            "🔍 Tìm kiếm công ty thoả mãn tiêu chí đánh giá"
        ],
        horizontal=True,
        key="sub_tab_3"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # --- NHÁNH 1: NHẬP ĐIỂM ĐÁNH GIÁ MỚI ---
    if sub_menu == "✍️ Tự nhập điểm đánh giá & Cập nhật Rating":
        all_company_names = df_companies["Company Name"].dropna().unique().tolist()
        eval_company = st.selectbox("1. Chọn doanh nghiệp bạn muốn đánh giá:", options=all_company_names)

        with st.form("manual_eval_form"):
            st.markdown("##### 2. Cảm nhận thực tế (Tuỳ chọn)")
            rev_title = st.text_input("Tiêu đề đánh giá:", placeholder="Ví dụ: Môi trường làm việc năng động, cơ hội thăng tiến tốt")
            rev_liked = st.text_area("Điều bạn hài lòng nhất:", placeholder="Ví dụ: Đồng nghiệp nhiệt tình, đãi ngộ tốt")
            rev_improve = st.text_area("Gợi ý cải thiện:", placeholder="Ví dụ: Cần bổ sung thêm phụ cấp học tập chứng chỉ")

            st.markdown("##### 3. Chấm điểm tiêu chí (Thang điểm 1.0 - 5.0)")
            col1, col2 = st.columns(2)
            with col1:
                r_rating = st.slider("Rating tổng quan:", 1.0, 5.0, 4.0, 0.5)
                r_salary = st.slider("Lương & Chế độ đãi ngộ:", 1.0, 5.0, 3.5, 0.5)
                r_training = st.slider("Đào tạo & Học tập:", 1.0, 5.0, 3.5, 0.5)
            with col2:
                r_management = st.slider("Sự quan tâm của Quản lý:", 1.0, 5.0, 4.0, 0.5)
                r_culture = st.slider("Văn hoá & Môi trường:", 1.0, 5.0, 4.0, 0.5)
                r_workspace = st.slider("Văn phòng & Không gian:", 1.0, 5.0, 4.0, 0.5)
                
            submitted_eval = st.form_submit_button("🤖 Phân Tích & Cập Nhật Đánh Giá", type="primary")

        if submitted_eval:
            input_features = pd.DataFrame([{
                "Rating": r_rating,
                "Salary & benefits": r_salary,
                "Training & learning": r_training,
                "Management cares about me": r_management,
                "Culture & fun": r_culture,
                "Office & workspace": r_workspace
            }])
            scaled_feats = scaler.transform(input_features)
            pred_res = classifier_model.predict(scaled_feats)[0]
            prob_res = classifier_model.predict_proba(scaled_feats)[0]
            auto_rec_label = "Yes" if pred_res == 1 else "No"

            st.markdown("---")
            if pred_res == 1:
                st.markdown(f"""
                <div style="background-color:rgba(22, 163, 74, 0.15); border:1px solid #16a34a; border-radius:16px; padding:20px; margin-bottom:16px;">
                    <h3 style="color:#22c55e; margin:0 0 8px 0;">✅ KẾT QUẢ: PHÙ HỢP / NÊN ĐỀ XUẤT (RECOMMEND: YES)</h3>
                    <p style="color:var(--text-color, #f8fafc); margin:0; font-size:14.5px;">
                        Mô hình AI dự đoán với độ tin cậy <b>{prob_res[1]*100:.1f}%</b> rằng doanh nghiệp <b>{eval_company}</b> sở hữu môi trường làm việc đạt chuẩn và được khuyến khích ứng tuyển.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:rgba(220, 38, 38, 0.15); border:1px solid #dc2626; border-radius:16px; padding:20px; margin-bottom:16px;">
                    <h3 style="color:#ef4444; margin:0 0 8px 0;">❌ KẾT QUẢ: CÂN NHẮC TRƯỚC KHI ỨNG TUYỂN (RECOMMEND: NO)</h3>
                    <p style="color:var(--text-color, #f8fafc); margin:0; font-size:14.5px;">
                        Mô hình AI dự đoán với độ tin cậy <b>{prob_res[0]*100:.1f}%</b> rằng các chỉ số hiện tại tại <b>{eval_company}</b> chưa đạt kỳ vọng của ứng viên.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Lưu vào file
            rev_file = "Reviews.xlsx"
            new_review_row = {
                "id": len(df_reviews) + 1,
                "Company Name": eval_company,
                "Cmt_day": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "Title": rev_title,
                "What I liked": rev_liked,
                "Suggestions for improvement": rev_improve,
                "Rating": r_rating,
                "Salary & benefits": r_salary,
                "Training & learning": r_training,
                "Management cares about me": r_management,
                "Culture & fun": r_culture,
                "Office & workspace": r_workspace,
                "Recommend?": auto_rec_label
            }

            try:
                updated_reviews = pd.concat([df_reviews, pd.DataFrame([new_review_row])], ignore_index=True)
                updated_reviews.to_excel(rev_file, index=False)
                st.success(f"💾 Đã lưu thành công đánh giá mới vào cơ sở dữ liệu (`{rev_file}`).")
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Lỗi khi lưu dữ liệu: {e}")

    # --- NHÁNH 2: TÌM KIẾM THỎA MÃN ĐÁNH GIÁ ---
    elif sub_menu == "🔍 Tìm kiếm công ty thoả mãn tiêu chí đánh giá":
        with st.form("filter_form"):
            st.markdown("##### Thiết lập ngưỡng điểm sàn mong muốn (Tối thiểu)")
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                min_rating = st.slider("Rating tổng quan:", 1.0, 5.0, 3.5, 0.1)
                min_salary = st.slider("Lương & Phúc lợi:", 1.0, 5.0, 3.0, 0.1)
            with f_col2:
                min_training = st.slider("Đào tạo & Phát triển:", 1.0, 5.0, 3.0, 0.1)
                min_management = st.slider("Quản lý quan tâm:", 1.0, 5.0, 3.0, 0.1)
            with f_col3:
                min_culture = st.slider("Văn hoá làm việc:", 1.0, 5.0, 3.5, 0.1)
                min_office = st.slider("Không gian văn phòng:", 1.0, 5.0, 3.0, 0.1)
            
            top_k_filter = st.slider("Số lượng công ty hiển thị:", 3, 20, 5)
            f_submitted = st.form_submit_button("🔍 Lọc Doanh Nghiệp Thỏa Mãn", type="primary")

        if f_submitted:
            condition = (
                (df_companies["Rating"] >= min_rating) &
                (df_companies["Salary & benefits"] >= min_salary) &
                (df_companies["Training & learning"] >= min_training) &
                (df_companies["Management cares about me"] >= min_management) &
                (df_companies["Culture & fun"] >= min_culture) &
                (df_companies["Office & workspace"] >= min_office)
            )
            filtered_df = df_companies[condition].copy()

            if filtered_df.empty:
                st.warning("Không tìm thấy công ty nào thoả mãn toàn bộ tiêu chí trên. Hãy hạ bớt điểm lọc!")
                st.session_state["filtered_df_display"] = pd.DataFrame()
            else:
                filtered_df = filtered_df.sort_values(by="Rating", ascending=False).head(top_k_filter)
                feats_filtered = filtered_df[num_features].values
                feats_scaled = scaler.transform(feats_filtered)
                filtered_df["Dự đoán ML"] = classifier_model.predict(feats_scaled)
                filtered_df["Dự đoán ML"] = filtered_df["Dự đoán ML"].map({1: "Recommend ✅", 0: "Not Recommend ❌"})

                st.session_state["filtered_companies_list"] = filtered_df["Company Name"].tolist()
                st.session_state["filtered_df_display"] = filtered_df

        if "filtered_df_display" in st.session_state and not st.session_state["filtered_df_display"].empty:
            df_show = st.session_state["filtered_df_display"]
            st.success(f"🎯 Đã tìm thấy **{len(df_show)}** công ty xuất sắc thỏa mãn tiêu chuẩn của bạn:")
            
            for _, r in df_show.iterrows():
                st.markdown(f"""
                <div class="company-card">
                    <div class="company-card-header">
                        <span class="company-name-title">{r['Company Name']}</span>
                        <span class="badge-pill badge-green">{r['Dự đoán ML']}</span>
                    </div>
                    <div style="display:flex; flex-wrap:wrap; gap:16px; font-size:13px; color:var(--text-color, #cbd5e1); margin-top:8px;">
                        <span>⭐ Rating: <b>{r['Rating']}</b></span>
                        <span>💰 Lương: <b>{r['Salary & benefits']}</b></span>
                        <span>📚 Đào tạo: <b>{r['Training & learning']}</b></span>
                        <span>🤝 Quản lý: <b>{r['Management cares about me']}</b></span>
                        <span>🎉 Văn hoá: <b>{r['Culture & fun']}</b></span>
                        <span>🏢 Văn phòng: <b>{r['Office & workspace']}</b></span>
                        <span style="color:#38bdf8; font-weight:700;">👍 {r['Actual_Recommend_Rate']}% khuyên làm việc</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Xem hồ sơ & Thống kê chi tiết")
            sel_filter_comp = st.selectbox("Chọn công ty để xem chi tiết toàn diện:", options=st.session_state["filtered_companies_list"])
            if sel_filter_comp:
                display_company_details(sel_filter_comp, df_companies)