import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. إعدادات الصفحة الاحترافية
st.set_page_config(
    page_title="AI Blood Cell Analyzer",
    page_icon="🔬",
    layout="centered"
)

# 2. تحميل موديل YOLO (العقل المدبر)
@st.cache_resource # لضمان تحميل الموديل مرة واحدة فقط عند فتح الموقع
def load_model():
    # تأكدي أن هذا الملف موجود في نفس مجلد الكود على جهازك
    model = YOLO("blood_expert_model.pt") 
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("تأكدي من وجود ملف blood_expert_model.pt بجانب هذا الملف.")

# 3. واجهة المستخدم الرسومية
st.title("🔬 Smart Blood Cell Analyzer")
st.markdown("""
مرحباً بكِ في نظام تحليل الدم الذكي.  
قم برفع صورة عينة الدم المجهرية، وسيقوم الذكاء الاصطناعي بتمييز الخلايا وعدّها تلقائياً.
""")

st.divider()

# 4. أداة رفع الصور (المكان الذي يضع فيه المستخدم صورته)
uploaded_file = st.file_uploader("اختر صورة عينة دم (JPG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- الخطوة السحرية: تحويل الصورة لـ RGB لضمان عملها بـ 3 قنوات ألوان فقط ---
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    # تقسيم الشاشة لعرض النتائج
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ الصورة الأصلية")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("🤖 تحليل الذكاء الاصطناعي")
        with st.spinner('جاري معالجة العينة...'):
            # إجراء عملية التنبؤ (Inference)
            results = model.predict(source=img_array, conf=0.25)
            
            # رسم المربعات والنتائج فوق الصورة
            res_plotted = results[0].plot()
            st.image(res_plotted, use_container_width=True)

    # 5. عرض تقرير النتائج النهائي بالأرقام (Metrics)
    st.divider()
    st.header("📊 تقرير المختبر الذكي")
    
    # استخراج قائمة الأصناف المكتشفة (0: Platelets, 1: RBC, 2: WBC)
    detected_classes = results[0].boxes.cls.cpu().numpy().tolist()
    
    # حساب الأعداد لكل نوع
    count_rbc = detected_classes.count(1)
    count_wbc = detected_classes.count(2)
    count_platelets = detected_classes.count(0)
    
    # عرض العدادات بشكل أنيق
    m1, m2, m3 = st.columns(3)
    m1.metric("خلايا حمراء (RBC)", count_rbc)
    m2.metric("خلايا بيضاء (WBC)", count_wbc)
    m3.metric("صفائح (Platelets)", count_platelets)

# تذييل الصفحة
st.sidebar.markdown("---")
st.sidebar.info("هذا التطبيق مخصص لأغراض تعليمية وبحثية، ويعتمد على تقنية YOLOv8 الحديثة للرؤية الحاسوبية.")
#لم اجربه