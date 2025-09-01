import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION & DATA ====================
node_catalog = {
    'Standard_D4plds_v6': {'mem_in_GB': 8, 'cores': 4, 'cost_per_hour': 0.228, 'description': 'Small, budget node'},
    'Standard_D8ds_v4': {'mem_in_GB': 32, 'cores': 8, 'cost_per_hour': 0.820, 'description': 'Medium general purpose'},
    'Standard_D16ds_v6': {'mem_in_GB': 64, 'cores': 16, 'cost_per_hour': 1.637, 'description': 'Large compute optimized'},
    'Standard_D16pds_v6': {'mem_in_GB': 64, 'cores': 16, 'cost_per_hour': 1.102, 'description': 'Large memory optimized'},
    'Standard_D64pls_v6': {'mem_in_GB': 128, 'cores': 64, 'cost_per_hour': 3.738, 'description': 'Extra-large, high-perf'},
    'Standard_D4ds_v6': {'mem_in_GB': 16, 'cores': 4, 'cost_per_hour': 0.433, 'description': 'Small, general purpose'},
}

job_types = [
    'customer_segmentation_ml', 'weekly_email_analytics', 'sales_funnel_analysis',
    'ad_performance_dashboard', 'user_behavior_analysis', 'real_time_web_analytics'
]

# ==================== ML MODEL TRAINING & CACHING ====================

@st.cache_data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = []
    
    # Define weights for CPU and Memory importance
    cpu_weight = 0.6
    mem_weight = 0.4
    
    for _ in range(num_samples):
        job_name = np.random.choice(job_types)
        node_type = np.random.choice(list(node_catalog.keys()))
        specs = node_catalog[node_type]
        
        # Runtime (minutes) - skewed distribution
        runtime = np.random.lognormal(3.2, 0.6)
        
        # Data volume depends on node size
        data_volume = np.random.gamma(6, specs['cores'] * 1.5)
        
        # CPU utilization patterns
        if specs['cores'] > 16:
            avg_cpu = np.random.uniform(15, 60)  # Larger nodes often underutilized
        else:
            avg_cpu = np.random.uniform(40, 95)  # Smaller nodes get more loaded
        
        # Memory utilization patterns
        if specs['mem_in_GB'] > 64:
            avg_mem = np.random.uniform(20, 70)
        else:
            avg_mem = np.random.uniform(40, 95)
        
        # Cost in $
        total_cost = (runtime / 60) * specs['cost_per_hour']
        
        # New optimization score:
        # - Weighted CPU + Memory utilization
        # - Normalized by cost
        # - Reward if more data processed per $ spent
        utilization_score = (cpu_weight * avg_cpu + mem_weight * avg_mem)
        efficiency_factor = (data_volume / (runtime + 1))  # throughput effect
        optimization_score = (utilization_score * efficiency_factor) / (total_cost + 1e-6)
        
        data.append({
            'job_name': job_name,
            'Runtime': round(runtime, 2),
            'data_volume(gb)': round(data_volume, 2),
            'node_type': node_type,
            'cores': specs['cores'],
            'mem_in_GB': specs['mem_in_GB'],
            'total_cost_consumption': round(total_cost, 2),
            'avg_cpu_utilization_perc': round(avg_cpu, 2),
            'avg_mem_used_perc': round(avg_mem, 2),
            'optimization_score': round(optimization_score, 2)
        })
    
    return pd.DataFrame(data)


@st.cache_resource
def train_model():
    df = generate_synthetic_data()

    # Features for the model - CORRECTED TO INCLUDE ALL RELEVANT INPUTS
    X = df[['job_name', 'Runtime', 'data_volume(gb)', 'cores', 'mem_in_GB', 'avg_cpu_utilization_perc', 'avg_mem_used_perc', 'node_type', 'total_cost_consumption']]
    y = df['optimization_score']

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(X[['job_name', 'node_type']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['job_name', 'node_type']))
    
    X = X.drop(columns=['job_name', 'node_type'])
    X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    return model, encoder, X.columns.tolist()

# ==================== RECOMMENDATION FUNCTION ====================
def get_recommendation(job_info, model, encoder, feature_names):
    recommendations = []
    
    # Predict the optimization score for all possible node types
    for node_type, specs in node_catalog.items():
        temp_input = job_info.copy()
        temp_input['cores'] = specs['cores']
        temp_input['mem_in_GB'] = specs['mem_in_GB']
        
        # Create a DataFrame with the correct features and values
        temp_df = pd.DataFrame([temp_input])
        
        # One-hot encode job_name and node_type
        encoded_features = encoder.transform(temp_df[['job_name', 'node_type']])
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['job_name', 'node_type']))
        
        # Combine numerical and encoded features
        X_temp = pd.concat([temp_df.drop(columns=['job_name', 'node_type']), encoded_df], axis=1)
        
        # Reindex to ensure feature order matches training data
        X_temp = X_temp.reindex(columns=feature_names, fill_value=0)
        
        predicted_score = model.predict(X_temp)[0]
        
        estimated_cost = (job_info['Runtime'] / 60) * specs['cost_per_hour']
        
        recommendations.append({
            "node_type": node_type,
            "predicted_score": predicted_score,
            "estimated_cost": estimated_cost,
            "description": specs["description"]
        })

    # Sort and return the top 5 recommendations
    recommendations.sort(key=lambda x: x["predicted_score"], reverse=True)
    return recommendations[:2]

# ==================== STREAMLIT UI ====================
st.title("Databricks Job Optimization Analyzer 🎯")
st.markdown("Analyze a completed job's performance to get recommendations for a more efficient cluster.")

# Train the model and get the encoder and feature names
model, encoder, feature_names = train_model()

# User input section
st.header("Enter Job Details for Analysis")
current_job_details = {}
with st.expander("Job Metrics", expanded=True):
    current_job_details['job_name'] = st.selectbox("Job Name", options=job_types, key="job_name_input")
    current_job_details['node_type'] = st.selectbox("Current Node Type", options=list(node_catalog.keys()), key="node_type_input")
    current_job_details['Runtime'] = st.number_input("Runtime (minutes)", min_value=1.0, value=60.0)
    current_job_details['data_volume(gb)'] = st.number_input("Data Volume (GB)", min_value=1.0, value=100.0)
    current_job_details['total_cost_consumption'] = st.number_input("Current Cost ($)", min_value=0.01, value=5.0)
    current_job_details['avg_cpu_utilization_perc'] = st.slider("CPU Utilization (%)", min_value=0.0, max_value=100.0, value=25.0)
    current_job_details['avg_mem_used_perc'] = st.slider("Memory Utilization (%)", min_value=0.0, max_value=100.0, value=30.0)

if st.button("Analyze Job & Get Recommendations"):
    recommendations = get_recommendation(current_job_details, model, encoder, feature_names)
    
    st.subheader("Analysis Results")
    
    best_recommendation = recommendations[0]
    
    # Determine if current node is the best
    if best_recommendation['node_type'] == current_job_details['node_type']:
        st.success(f"**The current cluster is the most efficient choice.** No changes recommended.")
    else:
        st.success(f"**Recommendation:** Switch to the recommended cluster for a better outcome.")
        savings = current_job_details['total_cost_consumption'] - best_recommendation["estimated_cost"]
       

    # Display the full list
    st.markdown("### Recommended Nodes:")
    table_data = []
    for i, rec in enumerate(recommendations):
        savings_or_cost_change = current_job_details['total_cost_consumption'] - rec['estimated_cost']
        table_data.append({
            "Rank": i + 1,
            "Node Type": rec['node_type'],
            "Description": rec['description'],
            "Estimated Cost": f"${rec['estimated_cost']:.2f}",
            
        })
    st.dataframe(pd.DataFrame(table_data).set_index('Rank'))