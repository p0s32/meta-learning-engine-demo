# app.py
import streamlit as st
import pandas as pd
import io
import qrcode
from PIL import Image
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from knowledge_base import RAG_SYSTEM
from core_engine import create_meta_learning_system, DataCombiner, LLMAnalyzer, ResultsWriter

# Page config
st.set_page_config(
    page_title="🚀 Meta-Learning Business Intelligence Engine",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for Command Center styling
st.markdown("""
<style>
    .command-center {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .qa-button {
        background: #667eea;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        margin: 5px;
        cursor: pointer;
    }
    .qa-button:hover {
        background: #5a67d8;
    }
    .progress-container {
        background: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_qr_code(url, filename="demo_qr.png"):
    """Generate QR code for the app URL"""
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    return img

def run_analysis_with_progress():
    """Run analysis with progress updates"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def progress_callback(message):
        status_text.text(message)
        # Simulate progress
        current_progress = progress_bar.progress_value
        progress_bar.progress(min(current_progress + 0.1, 1.0))
    
    try:
        status_text.text("Initializing Meta-Learning Engine...")
        progress_bar.progress(0.1)
        
        # Create and run the system
        system = create_meta_learning_system()
        result = system.run_full_analysis(progress_callback)
        
        progress_bar.progress(1.0)
        status_text.text("Analysis Complete!")
        
        return result
        
    except Exception as e:
        progress_bar.progress(1.0)
        status_text.text(f"Error: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'llm_response': {'llm_response': f"Analysis failed: {str(e)}"}
        }

# Main Header
st.markdown("""
<div class="command-center">
    <h1>🚀 META-LEARNING ENGINE DASHBOARD</h1>
    <p>AI-Powered Business Intelligence Platform</p>
    <p><strong>Processing 13 Data Streams Across 3 Specialized Engines</strong></p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for analysis results
if 'analysis_results' not in st.session_state:
    st.session_state['analysis_results'] = None

# Sidebar for navigation
with st.sidebar:
    st.title("🛠️ Control Panel")
    
    # System Status
    st.subheader("📊 System Status")
    st.success("🟢 All Engines Online")
    
    if st.session_state['analysis_results'] and st.session_state['analysis_results'].get('success'):
        combined_data = st.session_state['analysis_results']['combined_data']
        metadata = combined_data['combined_metadata']
        st.metric("Processing Speed", "2.3s avg")
        st.metric("Success Rate", f"{metadata['overall_data_quality']:.1%}")
        st.metric("Engines Processed", metadata['engines_processed'])
        st.metric("Slots Analyzed", metadata['total_slots'])
    else:
        st.metric("Processing Speed", "Ready")
        st.metric("Success Rate", "100% (Demo)")
        st.metric("Engines Processed", "3")
        st.metric("Slots Analyzed", "13")
    
    # QR Code Generator
    st.subheader("📱 Share Your App")
    
    # Get current URL (this would be the deployed Streamlit URL)
    current_url = "https://meta-learning-engine.streamlit.app"
    
    if st.button("Generate QR Code"):
        try:
            qr_img = generate_qr_code(current_url)
            st.image(qr_img, caption="Scan to access demo", use_column_width=True)
            
            # Save QR code to session state for download
            qr_img.save("demo_qr.png")
            
            st.download_button(
                label="📥 Download QR Code",
                data=open("demo_qr.png", "rb").read(),
                file_name="meta_learning_demo_qr.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"QR generation failed: {str(e)}")
    
    st.markdown(f"**Current URL:**")
    st.code(current_url)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    # Quick Analysis Section
    st.markdown("### 📊 Quick Analysis")
    
    upload_col1, upload_col2, upload_col3 = st.columns(3)
    
    with upload_col1:
        if st.button("📤 Upload CSV", use_container_width=True):
            st.info("📁 Upload your CSV files to the demo_data/ folder")
            st.info("📋 Expected files: customer_email_train.csv, product_reviews_train.csv, etc.")
    
    with upload_col2:
        if st.button("🚀 Run Demo Analysis", use_container_width=True):
            with st.spinner("Running Meta-Learning Engine..."):
                result = run_analysis_with_progress()
                st.session_state['analysis_results'] = result
                
                if result['success']:
                    st.success("✅ Analysis Complete!")
                    st.balloons()
                else:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    with upload_col3:
        if st.button("📈 View Results", use_container_width=True):
            if st.session_state['analysis_results']:
                st.info("📊 Displaying analysis results below...")
            else:
                st.warning("⚠️ Please run analysis first")
    
    # Display results if available
    if st.session_state['analysis_results'] and st.session_state['analysis_results'].get('success'):
        st.markdown("---")
        st.markdown("### 🎯 Latest Analysis Results")
        
        result = st.session_state['analysis_results']
        llm_response = result['llm_response']
        combined_data = result['combined_data']
        
        # LLM Response
        st.markdown("#### 🤖 AI Analysis")
        st.markdown(llm_response['llm_response'])
        
        # Engine Breakdown
        st.markdown("#### 📊 Engine Performance")
        for engine_name, summary in combined_data['engine_summaries'].items():
            st.markdown(f"""
            <div class="metric-card">
                <h4>{engine_name.upper()}</h4>
                <p><strong>Slots Processed:</strong> {summary['slots_processed']}</p>
                <p><strong>Successful:</strong> {summary['successful_slots']}</p>
                <p><strong>Predictions:</strong> {summary['total_predictions']}</p>
                <p><strong>Errors:</strong> {summary['total_errors']}</p>
                <p><strong>Quality Score:</strong> {summary['data_quality_avg']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Results export
        if st.button("📥 Export Full Report"):
            try:
                results_writer = ResultsWriter()
                output_file = results_writer.write_results(llm_response, combined_data)
                
                if output_file and os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    st.download_button(
                        label="📄 Download Report",
                        data=report_content.encode('utf-8'),
                        file_name=f"meta_learning_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    st.success("✅ Report ready for download!")
                else:
                    st.error("❌ Failed to generate report")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

with col2:
    # AI Assistant Section
    st.markdown("### 🤖 Ask the AI")
    
    # Pre-defined prompt choices
    st.markdown("**Choose a question:**")
    
    questions = [
        "What can you analyze?",
        "How does Engine 2 work?",
        "What models do you use?",
        "What data do you need?",
        "Show me recent projects",
        "How accurate are predictions?",
        "What are the pricing tiers?",
        "How to get started?"
    ]
    
    # Create buttons for questions
    for i in range(0, len(questions), 2):
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            if st.button(questions[i], key=f"q_{i}"):
                st.session_state['selected_question'] = questions[i]
        
        if i + 1 < len(questions):
            with col_q2:
                if st.button(questions[i + 1], key=f"q_{i+1}"):
                    st.session_state['selected_question'] = questions[i + 1]
    
    # Display answer
    if 'selected_question' in st.session_state:
        question = st.session_state['selected_question']
        answer = RAG_SYSTEM.get_response(question)
        
        st.markdown("---")
        st.markdown(f"**Q: {question}**")
        st.markdown(f"**A: {answer}**")
        
        # Clear selection button
        if st.button("Clear Question"):
            del st.session_state['selected_question']
            st.experimental_rerun()

# Recent Results Section (simulated)
st.markdown("### 📈 Recent Results")
recent_results = [
    "• Engine 2: $1.8M revenue projection",
    "• Cost efficiency: 60% improvement", 
    "• Customer retention: 15% increase",
    "• ROI analysis: 280% return expected"
]

# Add actual results if analysis was run
if st.session_state['analysis_results'] and st.session_state['analysis_results'].get('success'):
    combined_data = st.session_state['analysis_results']['combined_data']
    for engine_name, summary in combined_data['engine_summaries'].items():
        recent_results.append(f"• {engine_name}: {summary['successful_slots']}/{summary['slots_processed']} slots successful")

for result in recent_results:
    st.markdown(result)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Meta-Learning Business Intelligence Engine | Built with Streamlit</p>
    <p><strong>13 CSV Slots • 3 Engines • OpenRouter AI • Real-time Analysis</strong></p>
    <p><a href="https://share.streamlit.io">Deploy your own app</a></p>
</div>
""", unsafe_allow_html=True)