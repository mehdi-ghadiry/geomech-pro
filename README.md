# GeoMech Pro — معماری Backend/Frontend جدا

از این نسخه به بعد پروژه به دو سرویس مستقل تقسیم شده:

```
GeoMechanics_SaaS/
├── backend/                 # مغز محاسباتی (FastAPI) — بدون UI
│   ├── main.py               # اندپوینت‌های API
│   ├── geomechanics_core.py   # موتور 1D MEM (بدون تغییر منطقی)
│   ├── report_generator.py    # ساخت PDF
│   └── requirements.txt
├── frontend/                # رابط کاربری (Streamlit) — بدون منطق محاسباتی
│   ├── app.py                 # فقط UI، از طریق API صدا می‌زنه
│   ├── api_client.py          # لایه‌ی ارتباط HTTP با backend
│   └── requirements.txt
├── run_backend.bat
├── run_frontend.bat
└── README.md
```

## چرا این تفکیک مهمه؟

- **backend** هیچ وابستگی‌ای به Streamlit نداره. یعنی فردا می‌تونی همین backend رو
  زیر یک اپ موبایل، یک افزونه‌ی اکسل، یا یک مشتری دیگه هم قرار بدی — بدون این‌که
  یک خط از موتور محاسباتی رو دوباره بنویسی.
- **frontend** دیگه مستقیم به `geomechanics_core.py` دسترسی نداره؛ فقط از طریق
  `api_client.py` با backend حرف می‌زنه — دقیقاً مثل یک مشتری واقعی API.
- تست، دیپلوی و مقیاس‌دهی هرکدوم جدا از اون یکی ممکن می‌شه (مثلاً بعداً backend
  رو می‌تونی روی یک سرور قوی‌تر با چند worker بذاری، بدون این‌که به Streamlit دست بزنی).

## راه‌اندازی محلی (Windows)

## فرمت فایل ورودی

رابط کاربری و API از `LAS` استاندارد، `CSV`، `TXT` جدولی و `XLSX` پشتیبانی می‌کنند.
فایل‌های جدولی باید یک سطر عنوان داشته باشند و حداقل ستون‌های عمق، `DT` و `RHOB` را
شامل شوند؛ `DTS` اختیاری است. در مرحله‌ی بعد از آپلود، کاربر نام واقعی ستون‌ها را
خودش نگاشت می‌کند؛ بنابراین لازم نیست دقیقاً همین نام‌ها در فایل نوشته شده باشند.

پیش‌نیاز: یک virtual environment به اسم `venv` در ریشه‌ی پروژه که هم پکیج‌های
`backend/requirements.txt` و هم `frontend/requirements.txt` توش نصب شده باشن:

```bat
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
```

سپس در دو ترمینال جدا (یا با دوبار کلیک روی هر bat):

1. `run_backend.bat` → بک‌اند روی `http://localhost:8000` بالا میاد
   (مستندات خودکار API: `http://localhost:8000/docs`)
2. `run_frontend.bat` → داشبورد Streamlit روی `http://localhost:8501` بالا میاد
   و به‌صورت خودکار به backend محلی وصل می‌شه.

## دیپلوی روی دو سرور جدا

وقتی backend رو روی یک آدرس واقعی (مثلاً یک VPS یا Render/Railway) دیپلوی کردی،
فقط کافیه قبل از اجرای frontend این متغیر محیطی رو ست کنی تا به همون آدرس وصل بشه،
نه به `localhost`:

```bat
set GEOMECH_BACKEND_URL=https://api.yourdomain.com
streamlit run app.py
```

## نکته‌ی امنیتی مهم قبل از عمومی‌کردن

در `backend/main.py`، تنظیم CORS فعلاً `allow_origins=["*"]` هست (یعنی هر سایتی
می‌تونه به API درخواست بزنه) — این فقط برای توسعه‌ی محلی مناسبه. قبل از publish
کردن، این رو به آدرس واقعی frontend خودت محدود کن:

```python
allow_origins=["https://your-frontend-domain.com"]
```

## حساب کاربری و ذخیره‌سازی نتایج (جدید)

از این نسخه به بعد، backend یک دیتابیس SQLite داره (`backend/geomech.db`،
خودکار ساخته می‌شه، نیازی به نصب هیچ دیتابیس جداگانه‌ای نیست) که:

- **ثبت‌نام/ورود کاربر** رو مدیریت می‌کنه (پسورد با PBKDF2 هش می‌شه، سشن با JWT)
- هر کاربر می‌تونه بعد از محاسبه، نتایج یک چاه رو با یک اسم **ذخیره** کنه
- از سایدبار، زیر «My Saved Wells»، می‌تونه چاه‌های ذخیره‌شده‌ی خودش رو دوباره باز کنه یا حذف کنه

هر کاربر **فقط** چاه‌های خودش رو می‌بینه — این توی backend با فیلتر کردن هر
کوئری روی `user_id` تضمین می‌شه، نه فقط با مخفی‌کردن دکمه توی UI.

بدون لاگین هم (guest mode) همه‌چیز مثل قبل کار می‌کنه — فقط دکمه‌ی «Save»
در دسترس نیست.

**نکته‌ی امنیتی:** قبل از دیپلوی عمومی، حتماً `GEOMECH_SECRET_KEY` رو با یک
مقدار تصادفی و طولانی عوض کن (در `backend/auth.py` توضیح داده شده) — مقدار
پیش‌فرض فقط برای توسعه‌ی محلیه.
