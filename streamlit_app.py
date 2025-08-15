import streamlit as st
import pandas as pd 
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import shap
from sklearn.metrics import mean_squared_error


class DropMissingData(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.45):
        self. threshold = threshold
        self.columns_to_drop = []
        
        
    def fit(self, X, y=None):
        self.columns_to_drop = data.isnull().mean()[data.isnull().mean()>self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        X= X.drop(columns = self.columns_to_drop, errors='ignore')
        
        X=X.dropna()
        return X
    
class CategoricalEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.cat = []

    def fit(self, X, y=None):
        X = X.copy()
        self.cat = X.select_dtypes(include='object').columns
        self.encoders.fit(X[self.cat])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cat] = self.encoders.transform(X[self.cat])
        return X

class selectsignificant(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_features = []
        
    def fit(self,X,y):
        f_scores, p_values =f_regression(X, y)
        self.selected_features = X.columns[p_values<self.alpha].tolist()
        return self
    
    def transform(self,X):
        return X[self.selected_features]

class VIFCorrelationReducer(BaseEstimator, TransformerMixin):
    def __init__(self, corr_threshold=0.8, protected_columns=None):
        self.corr_threshold = corr_threshold
        self.protected_columns = protected_columns or []
        self.kept_features = []

    def fit(self, X, y=None):
        X = X.copy()
        self.kept_features = list(X.columns)

        while True:
            # Step 1: Correlation matrix
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Step 2: Identify correlated pairs
            to_consider = [
                (row, col)
                for col in upper.columns
                for row in upper.index
                if upper.loc[row, col] >= self.corr_threshold
                and row not in self.protected_columns
                and col not in self.protected_columns
            ]

            if not to_consider:
                break

            # Step 3: Compute VIFs
            vif = pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                index=X.columns
            )

            # Step 4: Drop one feature from each correlated pair based on higher VIF
            dropped = set()
            for f1, f2 in to_consider:
                if f1 in dropped or f2 in dropped:
                    continue  # skip if already dropped

                vif1 = vif.get(f1, np.inf)
                vif2 = vif.get(f2, np.inf)

                if vif1 > vif2:
                    drop = f1
                else:
                    drop = f2

                X = X.drop(columns=[drop])
                dropped.add(drop)

            # Step 5: Update list of features to keep
            self.features_to_keep = list(X.columns)

        return self

    def transform(self, X):
        return X[self.features_to_keep]

@st.cache_resource
def load_model_and_pipeline():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("full_pipeline.pkl")
    columns = joblib.load('saved_columns.pkl')
    return model, preprocessor, columns
    
st.set_page_config(page_title="House Price BI App", layout="wide")

st.title("🏠 House Price Prediction App")

section = st.sidebar.selectbox(
    "Select a section",
    ["Home", "Data Exploration", "Prediction", "Explainability", 'Simulation']
)
data_uncleaned = pd.read_csv('train.csv')
df_cleaned = pd.read_csv('df_cleaned.csv')


if section == "Home":
    st.markdown("## 🏠 House Price Intelligence App")

    st.markdown("""

 🏠 House Price Intelligence App

Welcome to your strategic edge in the housing market.  
This app is built for real estate developers, investors, analysts, and decision-makers looking to take their business to the next level.

Whether you're pricing properties, exploring market trends, or evaluating development opportunities, this platform helps you move with confidence and clarity.

🔍 **Explore Data** – Identify key patterns and trends driving the housing market.  
📈 **Predict Prices** – Estimate home values with a high-performing machine learning model.  
🧠 **Explain Results** – Understand exactly what factors influence each prediction using SHAP explanations.  
🎯 **Run Simulations** – Test "What-If" scenarios (e.g., increasing garage area by 20%) and instantly compare predictions to assess impact.

With this app, you don’t just track the market — you lead it.
    """)

    st.info("Use the dropdown in the menu to select a section and begin.")


#Data Exploration Page    
elif section=="Data Exploration":
    subpage= st.sidebar.radio('Select an Insights', 
                     ['Data Overview','Price Trends','Neighbourhood Prices', 'Correlation Heatmap', 'Missing Data'])
    if subpage == 'Data Overview':
        section1 = st.sidebar.selectbox('Select an option',
                                        ['Raw', 'Cleaned'])
        if section1 == 'Raw':
            st.markdown(f"**Rows** {data_uncleaned.shape[0]} | **Columns** {data_uncleaned.shape[1]}")
            st.write(data_uncleaned.head(10))
            col_info = pd.DataFrame({
                'Feature': data_uncleaned.columns,
                'Data type': data_uncleaned.dtypes.values
            })
            st.write(col_info)
        elif section1 == 'Cleaned':
            st.markdown(f"**Rows**{df_cleaned.shape[0]} | **Columns** {df_cleaned.shape[1]}")
            st.write(df_cleaned.head(10))
            col_info2 = pd.DataFrame({
                'Feature': df_cleaned.columns,
                'Data type': df_cleaned.dtypes.values
            })
            st.write(col_info2)
    
    elif subpage== 'Missing Data':
        st.subheader("Missing Data Heatmap")
        fig1, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data_uncleaned.isnull(), cbar=False, yticklabels=False, cmap='Blues', ax=ax)
        st.pyplot(fig1)
    
    elif subpage=='Correlation Heatmap':
        st.subheader('Correlation Heatmap')
        cor_df = df_cleaned.select_dtypes(include='number')
        corr = cor_df.corr()
        fig2 = px.imshow(corr, text_auto=True,color_continuous_scale='RdBu_r',
                        width=1000,   # Set width
                        height=800)
        st.plotly_chart(fig2)
        
    elif subpage == 'Price Trends':
        section2a = st.sidebar.selectbox('Select option',['Year Sold', 'Year Built'])
        if section2a == 'Year Sold':
            df1 = df_cleaned.groupby('YrSold')['SalePrice'].mean().reset_index()
            coeffs = np.polyfit(df1['YrSold'], df1['SalePrice'], deg=1)
            df1['trend'] = np.poly1d(coeffs)(df1['YrSold'])
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=df1['YrSold'], y=df1['SalePrice'], mode='lines+markers', name='Actual Price'))
            fig3.add_trace(go.Scatter(x=df1['YrSold'], y=df1['trend'], mode='lines', name='Trend Line'))
            fig3.update_layout(title='Yearly Trend of Average Sale Prices', xaxis_title='Year Sold', yaxis_title='Price',xaxis=dict(
                tickmode='linear',
                tick0=df1['YrSold'].min(),
                dtick=1,  # Show every year
                tickformat='.0f'))
            
            pce_iowa = pd.DataFrame({
                'Year': [2006, 2007, 2008, 2009, 2010],
                'PCE_Per_Capita_Iowa (USD)': [27856, 28819, 29761, 29602, 30456]
            })
            
            col1, col2 = st.columns([2,1])
            with col1:
                st.plotly_chart(fig3, use_container_width=True)

            with col2:
                st.write("### Per Capita Consumer Spending (Iowa)")
                st.dataframe(pce_iowa.set_index('Year'))
                st.caption(
                    "Source: U.S. Bureau of Economic Analysis (BEA) via "
                    "[FRED](https://fred.stlouisfed.org/series/IAPCEPC)")
            
        elif section2a == 'Year Built':
            df2 = df_cleaned.groupby('YearBuilt')['SalePrice'].mean().reset_index()
            coeffs2 = np.polyfit(df2['YearBuilt'], df2['SalePrice'], deg=1)
            df2['trend'] = np.poly1d(coeffs2)(df2['YearBuilt'])
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=df2['YearBuilt'], y=df2['SalePrice'], mode='lines+markers', name ='Average Price'))
            fig4.add_trace(go.Scatter(x=df2['YearBuilt'], y=df2['trend'], mode='lines',name ='Trend Line'))
            fig4.update_layout(title='Total Market Value of Homes by Build Year', xaxis_title='Year Built', yaxis_title='Price')
            st.plotly_chart(fig4)
    
    elif subpage=='Neighbourhood Prices':
        df3 = df_cleaned.groupby('Neighborhood')['SalePrice'].mean().sort_values().reset_index() 
        fig5 = px.bar(df3, x='Neighborhood', y='SalePrice', title = 'Average House Prices per Neighborhood')
        st.plotly_chart(fig5)


# Data Prediction page
elif section=='Prediction':
    model,preprocessor,columns =load_model_and_pipeline()
    section3a =st.sidebar.selectbox('Choose an option',['Batch Prediction', 'Single'])
    if section3a=='Single':
        with st.form('prediction form'):
            st.subheader('Enter House Details')
        
            overall_qual = st.select_slider("Overall Quality (1 = Very Poor, 10 = Excellent)",options=list(range(1, 11)))
            
            ExterQual = st.selectbox('Exterior Material Quality (Ex-Excellent, Gd-Good, TA-Typical/Average, FA- Fair, PO-Poor)', list(df_cleaned['ExterQual'].unique()))
            
            fullb = st.select_slider('No. of Full Bathrooms above ground', options=list(range(0,6)))
        
            ktch_qual = st.selectbox('Kitchen Quality(Ex-Excellent, Gd-Good, TA-Typical/Average, FA- Fair, PO-Poor)', list(df_cleaned['KitchenQual'].unique()))
            
            cent_a = st.radio('Central Air Conditioning',['Y', 'N'])
            
            yearbuilt = st.selectbox('Construction Year', sorted(list(df_cleaned['YearBuilt'].unique())))
            
            firep = st.select_slider('Number of Fireplaces', options=list(range(0,6)))
            
            totbsmtsf = st.select_slider('Total Basement Area (SQ)', options=list(range(100,6120)))
            
            TotRmsAbvGrd = st.select_slider('Total Rooms above ground (Ex. bathrooms) ', options=list(range(0,13)))        
            
            grg_are = st.select_slider('Garage Area', options=list(range(100,1500)))
        
        
            submit = st.form_submit_button("Predict")
        
        if submit:
            input_dict = {}

            for col in data_uncleaned.columns:
            # Fill user-provided inputs
                if col == 'OverallQual':
                    input_dict[col] = overall_qual
                elif col == 'ExterQual':
                    input_dict[col] = ExterQual
                elif col == 'FirePlaces':
                    input_dict[col] = firep
                elif col == 'KitchenQual':
                    input_dict[col] = ktch_qual
                elif col == 'GarageArea':
                    input_dict[col] = grg_are
                elif col == 'FullBath':
                    input_dict[col] = fullb
                elif col == 'YearBuilt':
                    input_dict[col] = yearbuilt
                elif col == 'TotalBsmtSF':
                    input_dict[col] = totbsmtsf
                elif col == 'CentralAir':
                    input_dict[col] = cent_a
                elif col == 'TotRmsAbvGrd':
                    input_dict[col] = TotRmsAbvGrd
                else:
                    if data_uncleaned[col].dtype == 'object':
                        input_dict[col] = data_uncleaned[col].mode()[0] 
                    else:
                        input_dict[col] = 0

            input_df = pd.DataFrame([input_dict])
            input_df = input_df.drop(columns='SalePrice')
            X_transformed = preprocessor[0][0].transform(input_df)
            X_transformed = X_transformed[columns]
            prediction = model.predict(X_transformed)
            new_pred = np.exp(prediction)
        
            st.success(f"Predicted Sale Price: ${new_pred[0]:,.2f}")
    
    elif section3a=='Batch Prediction':
        uploaded_files = st.file_uploader('Choose a file')
        if uploaded_files is not None:
            data= pd.read_csv(uploaded_files)
            
            missing_handler= DropMissingData(threshold=0.45)
            new = missing_handler.fit_transform(data)
            new = new.dropna()
            new1 =preprocessor[0][0].transform(new)
            new1 = new1[columns]
            prediction2 = model.predict(new1)
            new_pred2 = np.exp(prediction2)
            new1['PredictedPrice'] = new_pred2
            new1['PredictedPrice'] = new1['PredictedPrice'].apply(lambda x: f"${x:,.2f}")
            st.markdown(
                f"""
                <div style="background-color:#d4edda;padding:15px;border-radius:10px;margin-top:10px">
                <h4 style="color:#155724;">✅ Batch Predictions Completed</h4>
                </div>
                """,
                unsafe_allow_html=True)
            cols = ['PredictedPrice'] + [col for col in new1.columns if col != 'PredictedPrice']
            st.write(new1[cols])
            st.session_state.df = new1
            st.session_state.model  = model
            st.session_state.new_pred2 = new_pred2

#Model & Results Explanation            
elif section == 'Explainability':
    st.header("🔍 Model Explainability")

    st.markdown("""
    Use this section to understand how different features influence the model's predictions.
    Select an explanation type:
    """)

    explain_mode = st.radio("Choose explanation type", ['Global Feature Importance', 'Local Explanation (SHAP)'])


    model, preprocessor, columns = load_model_and_pipeline()

    if explain_mode == 'Global Feature Importance':
        st.subheader("📊 Top Features by Gain (XGBoost)")

        booster = model.get_booster()
        importance = booster.get_score(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(importance_df.head(20), x='Importance', y='Feature', orientation='h', title='Top 20 Feature Importances')
        st.plotly_chart(fig, use_container_width=True)
    
    elif explain_mode=='Local Explanation (SHAP)': 
        uploaded_files = st.file_uploader('Choose a file')
        if uploaded_files is not None:
            data= pd.read_csv(uploaded_files)
            
            missing_handler= DropMissingData(threshold=0.45)
            new = missing_handler.fit_transform(data)
            new = new.dropna()
            new1 =preprocessor[0][0].transform(new)
            new1 = new1[columns]
            prediction2 = model.predict(new1)
            row1 = st.number_input("Choose first instance", min_value=0, max_value=len(new1)-1, value=0)
            row2 = st.number_input("Choose second instance", min_value=0, max_value=len(new1)-1, value=1)
            if st.button("Explain Predictions"):
                row_df1 = new1.iloc[[row1]]
                row_df2 = new1.iloc[[row2]]
                explainer = shap.Explainer(model.predict, new1)

                shap_values = explainer(pd.concat([row_df1, row_df2]))
                base_price1 = np.exp(shap_values[0].base_values)
                shap_contribs1 = shap_values[0].values * base_price1

                base_price2 = np.exp(shap_values[1].base_values)
                shap_contribs2 = shap_values[1].values * base_price2
            
                explanation1 = shap.Explanation(
                    values=shap_contribs1,
                    base_values=base_price1,
                    data=row_df1.iloc[0],
                    feature_names=row_df1.columns
                )

                explanation2 = shap.Explanation(
                    values=shap_contribs2,
                    base_values=base_price2,
                    data=row_df2.iloc[0],
                    feature_names=row_df2.columns
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"Explanation for instance {row1}")
                    fig1, ax1 = plt.subplots()
                    shap.plots.waterfall(explanation1, max_display=10, show=False)
                    st.pyplot(fig1)

                with col2:
                    st.write(f"Explanation for instance {row2}")
                    fig2, ax2 = plt.subplots()
                    shap.plots.waterfall(explanation2, max_display=10, show=False)
                    st.pyplot(fig2)
        
elif section =='Simulation':
    df_copy = st.session_state.df.copy()
    baseline = st.session_state.df['PredictedPrice']
    df_copy.drop(columns = 'PredictedPrice', inplace=True)
    st.session_state.target = st.sidebar.selectbox("🎯 Select target variable", df_copy.columns)
    percent = st.sidebar.select_slider('Percentage increase', options = range(101))
    t_percent = 1+(percent/100)
    df_copy[st.session_state.target] = df_copy[st.session_state.target] * t_percent
    new_pred = st.session_state.model.predict(df_copy)
    new_pred = np.exp(new_pred)
    st.write(pd.DataFrame({
        'Base Prediction Prices': baseline,
        'Simulated Predicted Prices': [f"${x:,.2f}" for x in new_pred]
    }))
    st.write(f' There is an average difference of ${(np.sqrt(mean_squared_error(st.session_state.new_pred2,new_pred))):,.2f} when {st.session_state.target} was increased by {percent}%')
            
