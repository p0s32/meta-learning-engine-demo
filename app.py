# app.py - Meta-Learning Business Intelligence Command Center
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from knowledge_base import RAG_SYSTEM
    from core_engine import create_meta_learning_system, DataCombiner, LLMAnalyzer, ResultsWriter
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    # Fallback responses for demo
    RAG_SYSTEM = None

# Page config
st.set_page_config(
    page_title="🌟 Intelligence Command Center",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Enterprise Command Center
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header-welcome {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
    }
    .status-indicator {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 15px 0;
    }
    .chart-container {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .quick-actions {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .ai-prompt {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .ai-prompt:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .ai-response {
        background: #e8f4fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
    }
    .roadmap-item {
        background: white;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dropdown-content {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 3px solid #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Generate mock business data for realistic charts
def generate_business_data():
    # All-time data (2022-2024)
    dates_2022_2024 = pd.date_range('2022-01-01', '2024-12-31', freq='M')
    
    # Revenue data with growth trend
    base_revenue = 1800000
    revenue_growth = 0.035  # 3.5% monthly growth
    revenue_data = []
    for i, date in enumerate(dates_2022_2024):
        revenue = base_revenue * (1 + revenue_growth) ** i
        # Add some noise
        noise = 0.15 * revenue * (pd.random.random() - 0.5)
        revenue_data.append(max(revenue + noise, 1000000))
    
    # Customer retention data (improving trend)
    retention_start = 0.75
    retention_improvement = 0.015  # 1.5% improvement per quarter
    retention_data = []
    for i, date in enumerate(dates_2022_2024):
        quarter = i // 3
        retention = min(retention_start + (retention_improvement * quarter), 0.92)
        # Add noise
        noise = 0.02 * (pd.random.random() - 0.5)
        retention_data.append(max(min(retention + noise, 0.95), 0.70))
    
    # Processing efficiency data (improving)
    processing_start = 2.3
    processing_improvement = 0.1  # 0.1s improvement per quarter
    processing_data = []
    for i, date in enumerate(dates_2022_2024):
        quarter = i // 3
        processing_time = max(processing_start - (processing_improvement * quarter), 1.0)
        # Add noise
        noise = 0.3 * (pd.random.random() - 0.5)
        processing_data.append(max(processing_time + noise, 0.8))
    
    return {
        'revenue': pd.DataFrame({
            'date': dates_2022_2024,
            'revenue': revenue_data
        }),
        'retention': pd.DataFrame({
            'date': dates_2022_2024,
            'retention': retention_data
        }),
        'processing': pd.DataFrame({
            'date': dates_2022_2024,
            'processing_time': processing_data
        })
    }

# Generate prediction data for 2025
def generate_predictions():
    dates_2025 = pd.date_range('2025-01-01', '2025-12-31', freq='M')
    
    # Revenue predictions (following roadmap)
    base_2025 = 3200000  # Starting higher due to expansion
    revenue_pred = []
    for i, date in enumerate(dates_2025):
        # Accelerated growth due to European expansion and AI service
        monthly_growth = 0.045 if i < 6 else 0.055  # Faster growth in first half
        revenue = base_2025 * (1 + monthly_growth) ** i
        revenue_pred.append(max(revenue, 2000000))
    
    # Retention predictions (improved with AI customer service)
    retention_base = 0.85
    retention_pred = []
    for i, date in enumerate(dates_2025):
        improvement = 0.005 * (i // 3)  # 0.5% per quarter
        retention = min(retention_base + improvement, 0.92)
        retention_pred.append(retention)
    
    # Processing predictions (optimized)
    processing_base = 1.2
    processing_pred = []
    for i, date in enumerate(dates_2025):
        optimization = 0.05 * (i // 3)  # 0.05s improvement per quarter
        processing_time = max(processing_base - optimization, 0.8)
        processing_pred.append(processing_time)
    
    return {
        'revenue_pred': pd.DataFrame({
            'date': dates_2025,
            'revenue': revenue_pred
        }),
        'retention_pred': pd.DataFrame({
            'date': dates_2025,
            'retention': retention_pred
        }),
        'processing_pred': pd.DataFrame({
            'date': dates_2025,
            'processing_time': processing_pred
        })
    }

# AI Response Engine
def get_ai_response(question):
    if not RAG_AVAILABLE or not RAG_SYSTEM:
        # Pre-defined intelligent responses
        responses = {
            "📈 What's our biggest revenue opportunity this quarter?": 
            "Based on your Q4 data showing 127% target achievement, expanding to European markets presents a $3.2M opportunity with 89% confidence. Your customer retention analysis shows mobile-first users have 23% higher LTV. I recommend prioritizing European expansion and mobile app enhancement.",
            
            "🎯 Should we prioritize retention or acquisition?": 
            "Your data shows retention improving +3% per quarter naturally. However, new customer acquisition through European expansion offers $3.2M immediate revenue vs. $800K from retention optimization. Recommendation: 70% acquisition focus (European markets), 30% retention (AI customer service deployment).",
            
            "⚡ How can we reduce processing time by another 30%?": 
            "Current efficiency: 1.7s/slot (down from 2.3s in Jan 2024). To reach 30% reduction (1.2s target): 1) Deploy GPU acceleration in Q1 (40% improvement), 2) Optimize query algorithms by 15%, 3) Implement parallel processing (20% boost). Total projected: 1.1s/slot.",
            
            "🌍 Which market should we expand to next?": 
            "Analysis indicates European markets offer: €3.2M Q1 revenue opportunity, 89% confidence scoring, lower competition density (-32%), favorable regulatory environment. Asia-Pacific shows €1.8M potential but higher CAC. Recommendation: European expansion Q1, Asia-Pacific Q3 2025.",
            
            "🤖 What AI features would maximize customer value?": 
            "Customer behavior analysis reveals: 68% prefer mobile-first interactions, 45% abandon during complex processes, 73% value real-time insights. Priority AI features: 1) Predictive billing alerts, 2) Automated churn prevention, 3) Dynamic pricing optimization. Expected impact: +15% retention, +$1.2M annual revenue.",
            
            "📊 Show me the correlation between retention and revenue": 
            "Statistical analysis shows 0.87 correlation between retention rate and revenue growth. Each 1% retention improvement generates $240K additional annual revenue. Current retention: 87% (Q4), Revenue: $2.6M/month. Projected: 89% retention → $2.8M/month (+$2.4M annual)."
        }
        return responses.get(question, "I'm analyzing your latest data to provide insights on this question.")
    else:
        return RAG_SYSTEM.get_response(question)

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="header-welcome">🌟 INTELLIGENCE COMMAND CENTER • PREMIUM SUITE</div>
        <div class="header-subtitle">Welcome, Director Chen    🎯 Operations Dashboard • Live</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top 4-panel layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="status-indicator">
            <h3>🏠 HOME</h3>
            <p><strong>Personal Hub</strong></p>
            <p>• Quick Actions</p>
            <p>• Personal Metrics</p>
            <p>• Recent Activity</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Run Full Analysis", use_container_width=True):
            with st.spinner("Initializing Meta-Learning Engine..."):
                try:
                    system = create_meta_learning_system()
                    result = system.run_full_analysis()
                    st.success("✅ Analysis Complete!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="status-indicator" style="border-left-color: #28a745;">
            <h3>📊 REAL TIME</h3>
            <p><strong>Live Dashboard</strong></p>
            <p>Revenue: $2.4M ↗️15.2%</p>
            <p>Today's Users: 2,147</p>
            <p>Processing: 847 predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="status-indicator" style="border-left-color: #ffc107;">
            <h3>🎯 SMART INSIGHTS</h3>
            <p><strong>Analysis Hub</strong></p>
            <p>• Q4 Forecast: $8.9M (94.3%)</p>
            <p>• Customer Segments</p>
            <p>• Risk Models Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="status-indicator" style="border-left-color: #dc3545;">
            <h3>⚡ SYSTEM STATUS</h3>
            <p><strong>All Systems Operational</strong></p>
            <p>🟢 All 3 Engines Online</p>
            <p>🟢 Data Quality: 97%</p>
            <p>🔔 2 alerts pending</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quarter Predictions & Company Roadmap
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; margin-bottom: 20px; text-align: center; color: white;'>
        <h2>🎯 QUARTER PREDICTIONS & COMPANY ROADMAP</h2>
        <p style='font-size: 1.2rem; margin: 0;'>Q4 2024 Progress: 127% of target | Q1 2025 Quota: $11.2M</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate data
    business_data = generate_business_data()
    predictions = generate_predictions()
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue Chart
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Scatter(
            x=business_data['revenue']['date'],
            y=business_data['revenue']['revenue']/1000000,
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='#667eea', width=3),
            marker=dict(size=6)
        ))
        fig_revenue.add_trace(go.Scatter(
            x=predictions['revenue_pred']['date'],
            y=predictions['revenue_pred']['revenue']/1000000,
            mode='lines+markers',
            name='2025 Prediction',
            line=dict(color='#f5576c', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        fig_revenue.update_layout(
            title="Monthly Revenue: Historical vs 2025 Prediction",
            xaxis_title="Month",
            yaxis_title="Revenue ($ Millions)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Customer Retention Chart
        fig_retention = go.Figure()
        fig_retention.add_trace(go.Scatter(
            x=business_data['retention']['date'],
            y=business_data['retention']['retention']*100,
            mode='lines+markers',
            name='Historical Retention',
            line=dict(color='#28a745', width=3),
            marker=dict(size=6)
        ))
        fig_retention.add_trace(go.Scatter(
            x=predictions['retention_pred']['date'],
            y=predictions['retention_pred']['retention']*100,
            mode='lines+markers',
            name='2025 Prediction',
            line=dict(color='#ffc107', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        fig_retention.update_layout(
            title="Customer Retention Rate: Historical vs 2025 Prediction",
            xaxis_title="Quarter",
            yaxis_title="Retention Rate (%)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_retention, use_container_width=True)
    
    # Second row of charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Processing Efficiency Chart
        fig_processing = go.Figure()
        fig_processing.add_trace(go.Scatter(
            x=business_data['processing']['date'],
            y=business_data['processing']['processing_time'],
            mode='lines+markers',
            name='Historical Processing Time',
            line=dict(color='#dc3545', width=3),
            marker=dict(size=6)
        ))
        fig_processing.add_trace(go.Scatter(
            x=predictions['processing_pred']['date'],
            y=predictions['processing_pred']['processing_time'],
            mode='lines+markers',
            name='2025 Target',
            line=dict(color='#17a2b8', width=3, dash='dash'),
            marker=dict(size=6)
        ))
        fig_processing.update_layout(
            title="Processing Efficiency: Historical vs 2025 Target",
            xaxis_title="Month",
            yaxis_title="Processing Time (seconds)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_processing, use_container_width=True)
    
    with col4:
        # Key Metrics Summary
        st.markdown("""
        <div class="prediction-box">
            <h3>🎯 2025 Key Targets</h3>
            <p><strong>Revenue Goal:</strong> $45.6M/year</p>
            <p><strong>Retention Target:</strong> 89%</p>
            <p><strong>Processing Speed:</strong> <1.0s avg</p>
            <p><strong>European Expansion:</strong> €3.2M Q1</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Current Status</h4>
            <p><strong>Q4 Achievement:</strong> 127% of target</p>
            <p><strong>Growth Rate:</strong> +42% YoY</p>
            <p><strong>System Uptime:</strong> 99.7%</p>
            <p><strong>Data Quality:</strong> 97%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Actionable Roadmap
    st.markdown("---")
    st.markdown("### 🎯 ACTIONABLE ROADMAP & QUARTERLY GOALS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 BIG MOVES")
        
        # Big Moves with expandable details
        big_moves = [
            {
                'title': '🌍 Expand to European Markets',
                'target': 'Target: €3.2M Q1 revenue',
                'details': 'Phase 1: Germany, France, UK markets. Expected revenue: €1.1M/month by Q2. Requirements: Localization, compliance, local partnerships. Timeline: 8 weeks to full deployment.'
            },
            {
                'title': '🤖 Launch AI Customer Service',
                'goal': 'Goal: 60% ticket reduction',
                'details': 'Deploy conversational AI across all channels. Expected impact: 60% reduction in support tickets, 24/7 availability, 40% faster response times. Integration with existing CRM systems.'
            },
            {
                'title': '💳 Implement Predictive Billing',
                'roi': 'ROI: +25% cash flow',
                'details': 'AI-powered billing optimization based on usage patterns. Forecast accuracy improvement from 73% to 91%. Expected cash flow improvement: $2.4M annually.'
            }
        ]
        
        for i, move in enumerate(big_moves):
            with st.expander(f"{move['title']}"):
                st.markdown(f"""
                <div class="dropdown-content">
                    <p><strong>{list(move.keys())[1].title()}:</strong> {list(move.values())[1]}</p>
                    <p><strong>Details:</strong> {move['details']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔧 QUICK WINS")
        
        quick_wins = [
            {'task': '📊 Update dashboard analytics', 'eta': 'ETA: 2 weeks', 'details': 'Add real-time revenue tracking, customer segmentation insights, and predictive alerts. Estimated time savings: 4 hours/week.'},
            {'task': '⚡ Optimize database queries', 'eta': 'ETA: 1 week', 'details': 'Index optimization and query restructuring. Expected 30% performance improvement, reduced server load by 25%.'},
            {'task': '👥 Team training program', 'eta': 'ETA: 3 weeks', 'details': 'Advanced analytics and AI tools training for all team members. Expected productivity increase: 15%.'}
        ]
        
        for i, win in enumerate(quick_wins):
            with st.expander(f"{win['task']}"):
                st.markdown(f"""
                <div class="dropdown-content">
                    <p><strong>{win['eta']}</strong></p>
                    <p>{win['details']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # AI Chat Interface
    st.markdown("---")
    st.markdown("### 🤖 ASK THE INTELLIGENCE ENGINE")
    
    # Pre-defined AI prompts
    ai_prompts = [
        "📈 What's our biggest revenue opportunity this quarter?",
        "🎯 Should we prioritize retention or acquisition?",
        "⚡ How can we reduce processing time by another 30%?",
        "🌍 Which market should we expand to next?",
        "🤖 What AI features would maximize customer value?",
        "📊 Show me the correlation between retention and revenue"
    ]
    
    # Initialize session state for selected question
    if 'selected_ai_question' not in st.session_state:
        st.session_state.selected_ai_question = None
    
    # Display AI prompts as buttons
    cols_per_row = 2
    for i in range(0, len(ai_prompts), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, prompt in enumerate(ai_prompts[i:i+cols_per_row]):
            with cols[j]:
                if st.button(prompt, key=f"ai_prompt_{i+j}", use_container_width=True):
                    st.session_state.selected_ai_question = prompt
    
    # Display AI response
    if st.session_state.selected_ai_question:
        question = st.session_state.selected_ai_question
        response = get_ai_response(question)
        
        st.markdown(f"""
        <div class="ai-response">
            <h4>💬 {question}</h4>
            <p style="font-size: 1.1rem; line-height: 1.6;">{response}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("❌ Clear Question"):
            st.session_state.selected_ai_question = None
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌟 Intelligence Command Center • Premium Suite</p>
        <p><strong>13 Data Streams • 3 Specialized Engines • Real-time Analysis</strong></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
