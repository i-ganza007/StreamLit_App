import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from PIL import Image
import io

# ========== CONFIG ==========
BASE_URL = "https://shoe-type-classifier-summative.onrender.com"
st.set_page_config(page_title="Shoe Classifier Dashboard", layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
    }
    .css-1d391kg, .css-18e3th9 {
        color: white !important;
    }
    .css-ffhzg2 {
        background-color: rgba(0, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# ========== UTILS ==========
def fetch(endpoint):
    try:
        res = requests.get(f"{BASE_URL}/{endpoint}")
        if res.ok:
            return res.json()
        else:
            return {"error": f"HTTP {res.status_code}: {res.text[:200]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def post_file(file, endpoint):
    try:
        # Reset file pointer to beginning
        file.seek(0)
        res = requests.post(f"{BASE_URL}/{endpoint}/", files={"file": file})
        
        # Check if we got a valid response
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"HTTP {res.status_code}: {res.text[:200]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def post_zip_file(zip_file, endpoint):
    try:
        # Reset file pointer to beginning
        zip_file.seek(0)
        # Use 'zip_file' as the parameter name for the upload-zip endpoint
        res = requests.post(f"{BASE_URL}/{endpoint}/", files={"zip_file": zip_file})
        
        # Check if we got a valid response
        if res.status_code == 200:
            return res.json()
        else:
            return {"error": f"HTTP {res.status_code}: {res.text[:200]}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# ========== SIDEBAR ==========
st.sidebar.title("🔍 Navigation")
view = st.sidebar.radio("Go to", ["📊 Dashboard", "📷 Predict", "📁 Upload + Retrain"])

# ========== DASHBOARD ==========
if view == "📊 Dashboard":
    st.title("📊 System Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 System Metrics")
        metrics = fetch("metrics")
        if metrics and isinstance(metrics, list) and len(metrics) > 0:
            latest_metric = metrics[-1]
            
            # Main metrics row
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Total Requests", latest_metric.get('request_count', 'N/A'))
            with col_metric2:
                avg_time = latest_metric.get('avg_response_time', 0)
                display_time = round(avg_time, 2) if avg_time else 'N/A'
                st.metric("Avg Response Time (s)", display_time)
            with col_metric3:
                st.metric("Model Status", latest_metric.get('model_uptime_status', 'Unknown'))
            
            # Show recent metrics trend if we have multiple data points
            if len(metrics) > 1:
                st.subheader("📈 Response Time Trend")
                df_metrics = pd.DataFrame(metrics[-10:])  # Last 10 metrics
                if 'avg_response_time' in df_metrics.columns:
                    st.line_chart(df_metrics.set_index('timestamp')['avg_response_time'])
                    
        elif metrics and isinstance(metrics, dict) and metrics.get("error"):
            st.error(f"❌ Failed to load metrics: {metrics['error']}")
            st.info("💡 This usually means the backend server is down or sleeping.")
        else:
            st.warning("⚠️ No metrics data available")
            st.info("📊 Metrics will appear here once the backend starts receiving requests.")

    with col2:
        st.subheader("📈 Prediction Distribution")
        preds = fetch("predictions")
        if preds and isinstance(preds, list) and len(preds) > 0:
            df = pd.DataFrame(preds)
            
            if 'predicted_class' in df.columns:
                class_counts = df['predicted_class'].value_counts()
                
                # Create styled chart
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(class_counts.index, class_counts.values, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                
                # Styling
                ax.set_title("Predicted Class Distribution", color='white', fontsize=16)
                ax.set_xlabel("Shoe Class", color='white')
                ax.set_ylabel("Count", color='white')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom', color='white')
                
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                st.pyplot(fig)
                
                # Show summary stats
                st.info(f"📊 Total Predictions: {len(preds)} | Most Common: {class_counts.index[0]} ({class_counts.iloc[0]} times)")
                
                # Show recent predictions
                st.subheader("🔮 Recent Predictions")
                recent_preds = df.head(5)[['predicted_class', 'confidence_score', 'response_time']].copy()
                recent_preds['confidence_score'] = recent_preds['confidence_score'].apply(lambda x: f"{round(x*100, 1)}%")
                recent_preds['response_time'] = recent_preds['response_time'].astype(str) + " ms"
                st.dataframe(recent_preds, use_container_width=True)
            else:
                st.warning("⚠️ Predictions data format is unexpected")
                with st.expander("Debug Info"):
                    st.json(preds[:3])
                    
        elif preds and isinstance(preds, dict) and preds.get("error"):
            st.error(f"❌ Failed to load predictions: {preds['error']}")
            st.info("💡 This usually means the backend server is down or sleeping.")
        else:
            st.warning("⚠️ No prediction data available")
            st.info("🔮 Predictions will appear here once users start making predictions.")

    # Training Data Overview Section
    st.divider()
    st.subheader("📚 Training Data Overview")
    
    col_train1, col_train2 = st.columns(2)
    
    with col_train1:
        training_data = fetch("training-data")
        if training_data and isinstance(training_data, list) and len(training_data) > 0:
            df_training = pd.DataFrame(training_data)
            
            # Training data metrics
            total_training = len(training_data)
            processed_count = len(df_training[df_training['is_processed'] == True]) if 'is_processed' in df_training.columns else 0
            pending_count = total_training - processed_count
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("Total Training Images", total_training)
            with col_t2:
                st.metric("Processed", processed_count)
            with col_t3:
                st.metric("Pending", pending_count)
                
        else:
            st.info("📁 No training data uploaded yet")
    
    with col_train2:
        if training_data and isinstance(training_data, list) and len(training_data) > 0:
            if 'shoe_class' in df_training.columns:
                class_dist = df_training['shoe_class'].value_counts()
                
                # Training data class distribution
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.pie(class_dist.values, labels=class_dist.index, autopct='%1.1f%%', 
                       colors=['#FFB6C1', '#98FB98', '#87CEEB'])
                ax2.set_title("Training Data Class Distribution", color='white', fontsize=14)
                fig2.patch.set_facecolor('none')
                st.pyplot(fig2)
            else:
                st.info("📊 Upload training data to see class distribution")

# ========== PREDICT ==========
elif view == "📷 Predict":
    st.title("📷 Make a Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔮 Predict"):
            with st.spinner("Classifying..."):
                result = post_file(uploaded_file, "predict")
            
            if "predicted_class" in result:
                st.success(f"🎯 **Predicted Class: {result['predicted_class']}**")
                
                # Handle different confidence field names
                confidence = result.get('confidence_score') or result.get('confidence', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{round(confidence * 100, 2)}%")
                with col2:
                    if 'response_time' in result:
                        st.metric("Response Time", f"{result['response_time']} ms")
                
                # Show file URL if available
                if 'file_url' in result:
                    st.caption(f"📁 File stored at: {result['file_url']}")
                    
                # Technical details in expander
                with st.expander("🔍 Technical Details"):
                    st.json(result)
                    
            elif "error" in result:
                st.error(f"❌ Prediction failed: {result['error']}")
                
                # Provide helpful guidance based on error type
                error_msg = result["error"].lower()
                if "502" in result["error"] or "html" in error_msg:
                    st.warning("🚨 **Server Issue Detected**")
                    st.info("""
                    **Possible causes:**
                    - Render service is sleeping (free tier limitation)
                    - Server startup in progress
                    - Configuration issue
                    
                    **Solutions:**
                    1. Wait 30-60 seconds and try again
                    2. Check server status above
                    3. Try refreshing the page
                    """)
                    
                elif "timeout" in error_msg:
                    st.info("⏱️ **Timeout Error** - Server may be waking up from sleep")
                    
                elif "connection" in error_msg:
                    st.info("🔌 **Connection Error** - Check server status and URL")
            else:
                st.error("❌ Unexpected response format")
                st.json(result)

# ========== UPLOAD + RETRAIN ==========
else:
    st.title("📁 Upload Training Data & Retrain")

    zip_file = st.file_uploader("Upload a ZIP file of new training data", type="zip")
    if zip_file and st.button("⬆️ Upload ZIP"):
        with st.spinner("Uploading and processing data..."):
            res = post_zip_file(zip_file, "upload-zip")
            
            if res.get("error"):
                st.error(f"❌ Upload failed: {res['error']}")
            else:
                st.success("✅ ZIP file uploaded successfully!")
                st.json(res)

    st.divider()

    if st.button("♻️ Retrain Model"):
        with st.spinner("Retraining in progress..."):
            res = fetch("retrain")
            st.success("Retraining complete.")
            st.json(res)

    # Training data status
    st.divider()
    st.subheader("📊 Training Data Status")
    
    col_status1, col_status2 = st.columns([1, 1])
    
    with col_status1:
        if st.button("📊 Check Training Data"):
            with st.spinner("Fetching training data info..."):
                training_data = fetch("training-data")
            
            if training_data and isinstance(training_data, list) and len(training_data) > 0:
                df = pd.DataFrame(training_data)
                
                st.success(f"📊 Found {len(training_data)} training records")
                
                # Show summary statistics
                if 'shoe_class' in df.columns:
                    class_summary = df['shoe_class'].value_counts()
                    
                    # Create summary metrics
                    col_sum1, col_sum2, col_sum3 = st.columns(3)
                    with col_sum1:
                        st.metric("Boot Images", class_summary.get('Boot', 0))
                    with col_sum2:
                        st.metric("Sandal Images", class_summary.get('Sandal', 0))  
                    with col_sum3:
                        st.metric("Shoe Images", class_summary.get('Shoe', 0))
                
                # Show processing status
                if 'is_processed' in df.columns:
                    processed_count = len(df[df['is_processed'] == True])
                    pending_count = len(df[df['is_processed'] == False])
                    
                    st.subheader("Processing Status")
                    col_proc1, col_proc2 = st.columns(2)
                    with col_proc1:
                        st.metric("Processed", processed_count, help="Images used for training")
                    with col_proc2:
                        st.metric("Pending", pending_count, help="Images waiting to be processed")
                
            elif training_data and isinstance(training_data, dict) and training_data.get("error"):
                st.error(f"❌ Failed to fetch training data: {training_data['error']}")
            else:
                st.warning("⚠️ No training data found")
                st.info("📁 Upload some training data using the ZIP upload feature above")
    
    with col_status2:
        if st.button("📈 View Recent Training Data"):
            with st.spinner("Loading recent uploads..."):
                training_data = fetch("training-data")
            
            if training_data and isinstance(training_data, list) and len(training_data) > 0:
                df = pd.DataFrame(training_data)
                
                # Show recent entries table
                st.subheader("Recent Training Uploads")
                display_cols = ['shoe_class', 'is_processed', 'uploaded_at']
                available_cols = [col for col in display_cols if col in df.columns]
                
                if available_cols:
                    recent_df = df[available_cols].head(10).copy()
                    if 'uploaded_at' in recent_df.columns:
                        recent_df['uploaded_at'] = pd.to_datetime(recent_df['uploaded_at']).dt.strftime('%Y-%m-%d %H:%M')
                    if 'is_processed' in recent_df.columns:
                        recent_df['is_processed'] = recent_df['is_processed'].map({True: '✅ Yes', False: '⏳ Pending'})
                    
                    st.dataframe(recent_df, use_container_width=True)
                else:
                    st.dataframe(df.head(10), use_container_width=True)
            else:
                st.info("📊 No training data to display")
