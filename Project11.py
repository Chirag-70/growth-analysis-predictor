import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score ,mean_absolute_error
import altair as alt
import requests




st.set_page_config(layout="wide")
st.title("Currently and upcoming growing sector prediction/analysis")
st.markdown(":material/home: Welcome to my app!")
selected_option = st.selectbox("Choose Who are YOU?:",["Student","Employee","Entrepreneur","Researcher","Other/Curious"],index=None,placeholder="I AM THE..")
st.caption(f"So you are the {selected_option}")
if selected_option=="Student":
    st.write("Great you are in the right way")
    st.toast('Successfully choosed Student!', icon='🎉')
elif selected_option=="Employee":
    st.write("Great,Means you want upgrade/Gain more insights") 
    st.toast('Successfully choosed Employee!', icon='🎉')   
elif selected_option=="Entrepreneur":
    st.write("Specially,You are the person ,for which i made this ninth Wonder")  
    st.toast('Successfully choosed Entrepreneur!', icon='🎉')  
elif selected_option=="Researcher":
    st.write("Reason doesn't specify your work, but Knowing everything is a blessing")
else:
    st.write("Explore Buddy...")      
with st.sidebar:
    st.write(":spiral_calendar:")
    with st.expander("Note for YOU*!"):
            st.success("Welcome to our APP buddy, In which we analyze the current,upcoming and 8 core industries contribution to GDP and IIP so ,we can study them to find out crucial insights..")

    st.info(f"Character:{selected_option}")    
         



# Data collection
df1 = pd.read_csv("Electricity.csv")
df2 = pd.read_csv("steel1.csv")
df3 = pd.read_csv("Refinery_Products.csv")
df4 = pd.read_csv("Coal.csv")
df5 = pd.read_csv("Crude oil.csv")
df6 = pd.read_csv("Natural Gas.csv")
df7 = pd.read_csv("cement.csv")
df8 = pd.read_csv("Fertilizers.csv")

# Visualization
cols1, cols2 = st.columns(2)

with cols1:
    st.header("Here are the 8 CORE PILLARS of Indian Economy")
    st.write("Covers 40% of :india: Economy (IIP)")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9=st.tabs(["Electricity","Steel","Refinery Products","Coal","Crude oil","Natural gas","Cement","Fertilizers","Distribution"])
    ######################
    with tab9:
        tabA,tabB=st.tabs(["Pie Chart Visualisation","Tabular Visualisation"])
        with tabA:
            st.title("Currently and Upcoming Growing Sectors--")
            st.header("Individual Contribution")
            X1=["Refinery Products","Electricity","Steel","Coal","Crude Oil","Natural Gas","Cement","Fertilizers"]
            Y1=[28.4,19.85,17.92,10.33,8.98,6.88,5.37,2.63]
            colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','grey','red','white','yellow']
            fig = plt.figure(figsize=(8,6))

            plt.pie(Y1, labels=X1, colors=colors, autopct='%1.1f%%',shadow=True, startangle=90)
            plt.axis("equal")  
            st.pyplot(fig)
        with tabB:
            data ={"Industries":["Refinery Products","Electricity","Steel","Coal","Crude Oil","Natural Gas","Cement","Fertilizers"],
                   "Contribution_in_IIP":[28.4,19.85,17.92,10.33,8.98,6.88,5.37,2.63]
                   }
            st.table(data)
           
    #######################################
    # ELECTRICITY..
    with tab1:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Electricity"):
                st.success("Electricity produced between 2016 to 2019")
            df1["Year"] = df1["Year"].astype(str)
            df_plot = df1.set_index("Year")
            st.bar_chart(df_plot["Production"])
            st.info("Electricity production are in Killowatt Hour")

            with st.expander("Want to predict future electricity production.."):
                Model_SElectioN = st.selectbox("Choose your model:",["Linear Regression","Random Forest Regressor"],index=None,placeholder="Enter Model name(eligible)..")
                st.caption(f"So you choose the Model-- {Model_SElectioN}")
                if Model_SElectioN == "Linear Regression":
                    input1 = st.number_input("Enter the year e.g. 2017", min_value=2015, max_value=2047, step=1)
                    if input1 != 0:
                        df1["Year_num"] = df1["Year"].str[:4].astype(int)

                        X = df1[["Year_num"]]
                        y = df1["Production"]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        model = LinearRegression()
                        model.fit(X_train, y_train)

                    
                        y_pred_single = model.predict([[int(input1)]])
                        
                        st.success(f"Predicted Electricity Production for {int(input1)} = {int(y_pred_single[0])} Kilowatt hour")

                        y_pred_test = model.predict(X_test)

                        r2 = r2_score(y_test, y_pred_test)
                        mse = mean_squared_error(y_test, y_pred_test)
                        rmse = np.sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred_test)

                        st.subheader(" Model Accuracy Metrics")
                        colst1,colst2,colst3,colst4=st.columns(4)
                        with colst1:
                            st.metric("R² Score", f"{r2:.4f}")

                        with colst2:
                            st.metric("MSE", f"{mse:.2f}")

                        with colst3:
                            st.metric("RMSE", f"{rmse:.2f}")

                        with colst4:
                            st.metric("MAE", f"{mae:.2f}")

                    else:
                        st.warning("Please enter a valid year (not 0).")
                                    
                elif Model_SElectioN=="Random Forest Regressor":
                    df1["Year"] = df1["Year"].str.split("-").str[1].astype(int)
                    df1["Year"] = df1["Year"] + 2000  


                    X = df1[["Year"]]
                    y = df1["Production"]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    model = RandomForestRegressor(n_estimators=200, random_state=42)
                    model.fit(X_train, y_train)

                    y_pred = model.predict(X_test)
                    st.markdown("---")
                    st.subheader(" Predict Future Production")

                    future_year = st.number_input(
                        "Enter a future year (e.g., 2026, 2030, 2035)",
                        min_value=2010,
                        max_value=2047,
                        step=1
                    )

                    if st.button("Predict"):
                        future_df = pd.DataFrame({"Year": [future_year]})
                        future_pred = model.predict(future_df)[0]
                        st.success(f"**Predicted Production in {future_year}: {future_pred:.2f} killowatt hour")
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    col1.metric(" Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
                    col2.metric(" R² Score", f"{r2_score(y_test, y_pred):.4f}")    
                    st.markdown("---")
                else:
                    pass
                

            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                start = df1["Production"].iloc[0]
                end = df1["Production"].iloc[-1]
                years = len(df1) - 1

                cagr = (end / start) ** (1 / years) - 1
                st.success(f"CAGR: {round(cagr * 100, 2)}%")    
        with tabB:
            df1["Production"] = pd.to_numeric(df1["Production"], errors="coerce")
            st.title("Electricity Production (2010–2025)")
            st.line_chart(
                df1,
                x="Year",
                y="Production",
                use_container_width=True
            )

            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df1["Production"].max()
                Min_production=df1["Production"].min()
                Median_production=df1["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)
               

    #######################################
    # STEEL..
    with tab2:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Steel"):
                st.success("Steel produced/Processed between 2016 to 2019")
            df2["Year"] = df2["Year"].astype(str)
            df_plot1 = df2.set_index("Year")
            st.bar_chart(df_plot1["Crude Steel Production"])
            st.info("Steel production are in Metric tonnes")

            with st.expander("Want to predict future Steel production.."):
                input2 = st.number_input("Enter the year e.g. 2015", min_value=2015, max_value=2047, step=1)
                if input2 != 0:
                    df2["Year_num1"] = df2["Year"].str[:4].astype(int)
                    X = df2[["Year_num1"]]
                    y = df2["Crude Steel Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input2)]])
                    st.success(f"Predicted Electricity Production for {int(input2)} = {int(y_pred[0])} Metric tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                start = df2["Crude Steel Production"].iloc[0]
                end = df2["Crude Steel Production"].iloc[-1]
                years = len(df2) - 1

                cagr = (end / start) ** (1 / years) - 1
                st.success(f"CAGR: {round(cagr * 100, 2)}%")      
        with tabB:
            df2["Crude Steel Production"] = pd.to_numeric(df2["Crude Steel Production"], errors="coerce")

            st.title("Crude Steel Production (2015–2024)")


            st.line_chart(
                    df2,
                    x="Year",
                    y="Crude Steel Production",
                    use_container_width=True
             )


             
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df2["Crude Steel Production"].max()
                Min_production=df2["Crude Steel Production"].min()
                Median_production=df2["Crude Steel Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)
    #######################################
    # REFINERY..
    with tab3:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Refinery Products"):
                st.success("Refinery Products produced between 2010 to 2024")
            df3["Year"] = df3["Year"].astype(str)
            df_plot2 = df3.set_index("Year")
            st.bar_chart(df_plot2["Refinery Production"])
            st.info("Refinery products production are in Million Metric tonnes")

            with st.expander("Want to predict future Refinery products production.."):
                input3 = st.number_input("Enter the year e.g. 2015", min_value=2014, max_value=2047, step=1)
                if input3 != 0:
                    df3["Year_num2"] = df3["Year"].str[:4].astype(int)
                    X = df3[["Year_num2"]]
                    y = df3["Refinery Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input3)]])
                    st.success(f"Predicted Electricity Production for {int(input3)} = {int(y_pred[0])} Million Metric tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df3["Refinery Production"].iloc[0]
                    end = df3["Refinery Production"].iloc[-1]
                    years = len(df3) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")       
        with tabB:
          

            st.title("Refinery Production Over Years")
            df3 = pd.read_csv("Refinery_Products.csv")
            st.line_chart(df3.set_index('Year'))
            st.caption("Year")

            
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df3["Refinery Production"].max()
                Min_production=df3["Refinery Production"].min()
                Median_production=df3["Refinery Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)

    #######################################
    # COAL..
    with tab4:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Coal"):
                st.success("Coal produced between 2010 to 2024")
            df4["Year"] = df4["Year"].astype(str)
            df_plot3 = df4.set_index("Year")
            st.bar_chart(df_plot3["Production"])
            st.info("Coal production are in Million tonnes")

            with st.expander("Want to predict future Refinery products production.."):
                input4 = st.number_input("Enter the year e.g. 2013", min_value=2014, max_value=2047, step=1)
                if input4 != 0:
                    df4["Year_num3"] = df4["Year"].str[:4].astype(int)
                    X = df4[["Year_num3"]]
                    y = df4["Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input4)]])
                    st.success(f"Predicted Electricity Production for {int(input4)} = {int(y_pred[0])} Million tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
                
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df4["Production"].iloc[0]
                    end = df4["Production"].iloc[-1]
                    years = len(df4) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")           
        with tabB:
            st.title("Coal Production from 2010 TO 2024")
            df4 = pd.read_csv("Coal.csv")
            st.line_chart(df4.set_index('Year'))
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df4["Production"].max()
                Min_production=df4["Production"].min()
                Median_production=df4["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)

    #######################################
    # CRUDE OIL..
    with tab5:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Crude oil"):
                st.success("Crude oil produced between 2010 to 2024")
            df5["Year"] = df5["Year"].astype(str)
            df_plot4 = df5.set_index("Year")
            st.bar_chart(df_plot4["Production"])
            st.info("Crude oil production are in Million Metric tonnes")

            with st.expander("Want to predict future Refinery products production.."):
                input5 = st.number_input("Enter the year e.g. 2012", min_value=2014, max_value=2047, step=1)
                if input5 != 0:
                    df5["Year_num4"] = df5["Year"].str[:4].astype(int)
                    X = df5[["Year_num4"]]
                    y = df5["Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input5)]])
                    st.success(f"Predicted Electricity Production for {int(input5)} = {int(y_pred[0])} Million Metric tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df5["Production"].iloc[0]
                    end = df5["Production"].iloc[-1]
                    years = len(df5) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")          
        with tabB:
           st.title("Crude Oil Production from 2010 TO 2024")
           df5 = pd.read_csv("Crude oil.csv")
           st.line_chart(df5.set_index('Year'))
        with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df5["Production"].max()
                Min_production=df5["Production"].min()
                Median_production=df5["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)
    #######################################
    # NATURAL GAS..
    with tab6:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Natural Gas"):
                st.success("Natural Gas produced between 2010 to 2024")
            df6["Year"] = df6["Year"].astype(str)
            df_plot5 = df6.set_index("Year")
            st.bar_chart(df_plot5["Production"])
            st.info("Crude oil production are in Billion Cubic Metres")

            with st.expander("Want to predict future Refinery products production.."):
                input6 = st.number_input("Enter the year e.g. 2011", min_value=2014, max_value=2047, step=1)
                if input6 != 0:
                    df6["Year_num5"] = df6["Year"].str[:4].astype(int)
                    X = df6[["Year_num5"]]
                    y = df6["Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input6)]])
                    st.success(f"Predicted Electricity Production for {int(input6)} = {int(y_pred[0])} Billion Cubic Metres")
                else:
                  st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df6["Production"].iloc[0]
                    end = df6["Production"].iloc[-1]
                    years = len(df6) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")     
        with tabB:
            st.title("Natural Gas Production from 2010 TO 2024")
            df6 = pd.read_csv("Natural Gas.csv")
            st.line_chart(df6.set_index('Year'))
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df6["Production"].max()
                Min_production=df6["Production"].min()
                Median_production=df6["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)
    #######################################
    # CEMENT..
    with tab7:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Cement"):
                st.success("Cement produced between 2010 to 2024")

            df7["Year"] = df7["Year"].astype(str)
            df_plot6 = df7.set_index("Year")

            st.bar_chart(df_plot6["Production"])
            st.info("Cement production are in Million Tonnes")

            with st.expander("Want to predict future Refinery products production.."):
                input7 = st.number_input("Enter the year e.g. 2010", min_value=2014, max_value=2047, step=1)
                if input7 != 0:
                    df7["Year_num6"] = df7["Year"].str[:4].astype(int)   # FIX 2
                    X = df7[["Year_num6"]]
                    y = df7["Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input7)]])
                    st.success(f"Predicted Electricity Production for {int(input7)} = {int(y_pred[0])} Million Tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df7["Production"].iloc[0]
                    end = df7["Production"].iloc[-1]
                    years = len(df7) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")         
        with tabB:
            st.title("Cement Production from 2010 TO 2024")
            df7= pd.read_csv("cement.csv")
            st.line_chart(df7.set_index('Year'))
            st.caption("Year")
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df7["Production"].max()
                Min_production=df7["Production"].min()
                Median_production=df7["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)

    #######################################
    # FERTILIZERSs..
    with tab8:
        tabA,tabB=st.tabs(["Bar Chart Visualization","Line Chart"])
        with tabA:
            with st.expander("Fertlizers"):
                st.success("Fertilizers produced between 2010 to 2024")

            df8["Year"] = df8["Year"].astype(str)
            df_plot7 = df8.set_index("Year")

            st.bar_chart(df_plot7["Production"])
            st.info("Fertilizers production are in Lakh Metric Tonnes")

            with st.expander("Want to predict future Refinery products production.."):
                input8 = st.number_input("Enter the year e.g. 2019", min_value=2014, max_value=2047, step=1)
                if input8 != 0:
                    df8["Year_num7"] = df8["Year"].str[:4].astype(int)   # FIX 2
                    X = df8[["Year_num7"]]
                    y = df8["Production"]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict([[int(input8)]])
                    st.success(f"Predicted Electricity Production for {int(input8)} = {int(y_pred[0])} Lakh Metric Tonnes")
                else:
                    st.warning("Please enter a valid year (not 0).")
            with st.expander("Calculate [Compound Anual Growth Rate] CAGR"):
                    start = df8["Production"].iloc[0]
                    end = df8["Production"].iloc[-1]
                    years = len(df8) - 1

                    cagr = (end / start) ** (1 / years) - 1
                    st.success(f"CAGR: {round(cagr * 100, 2)}%")      
        with tabB:
            st.title("Fertilizers from 2010 TO 2024")
            df8= pd.read_csv("Fertilizers.csv")
            st.line_chart(df8.set_index('Year'))
            with st.expander("Max.,Min.,Median---->Productions.."):
                Max_production=df8["Production"].max()
                Min_production=df8["Production"].min()
                Median_production=df8["Production"].median()
                st.write(Max_production)
                st.write(Min_production)
                st.write(Median_production)

st.divider()
st.divider()

st.title("🌍 Live GDP Tracker")

country_name = st.text_input("Enter Country Name (e.g., India, USA, Germany):", "India")

def fetch_gdp_data(country):
    """Fetch GDP data from API and return DataFrame"""
    url = f"https://api.api-ninjas.com/v1/gdp?country={country}"
    headers = {"X-Api-Key": "doExqX7rVmpK00R8xpKipA==VobrzbfhJMjEeUaW"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            st.warning(f"No GDP data found for '{country}'")
            return None
        df = pd.DataFrame(data)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None
    except ValueError:
        st.error("Failed to parse JSON response")
        return None

df = fetch_gdp_data(country_name)
if df is not None:
    st.subheader(f"Raw GDP Data for {country_name}")
    st.dataframe(df)

    
    def plot_gdp_chart():
        required_cols = ['country', 'year', 'gdp_growth']
        if not all(col in df.columns for col in required_cols):
            st.error(f"API response does not contain required columns: {required_cols}")
            return
        df_chart = df[required_cols]
        df_chart = df_chart[(df_chart['year'] >= 1980) & (df_chart['year'] <= 2029)]
        df_chart['country'] = df_chart['country'].str[:3].str.upper()
        st.success(f"GDP Growth of {country_name} ({df_chart['country'].iloc[0]}) from 1980 to 2029")
        st.line_chart(df_chart.set_index('year')['gdp_growth'])


    st.button(
        label="Get GDP Line Chart",
        key="gdp_chart_button",
        help="Click to see GDP growth line chart",
        on_click=plot_gdp_chart,
        type="primary",
        icon="💹",
        use_container_width=True
    )



with cols2:
    st.header("Sectors")
    tab1,tab2,tab3=st.tabs(["Current Growing sectors","Upcoming Growing sectors","Analyze Individually"])
    with tab1:
        st.info("For better experience view chart in Full screen :material/open_in_full:  and more REarranging chart use Autoscale")
        with st.expander("Query"):
            st.write("Data represents GDP_contribution in %")

        hf1=pd.read_csv("current_sectors.csv")
        fig = px.scatter(
        hf1,
        x="Sector",
        y="GDP_Contribution_%",
        size="GDP_Contribution_%",
        color="Specification",
        hover_name="Sector",
        size_max=90,              
        width=1600,               
        height=500,              
        title="Current Sectors Bubble Chart",
    )

        fig.update_layout(
            xaxis_title="Sectors",
            yaxis_title="GDP Contribution (%)",
            font=dict(size=18),       
        )

        
        st.plotly_chart(fig, use_container_width=True)
        ##################################
    with tab2:
        st.info("For better experience view chart in Full screen :material/open_in_full:")
        with st.expander("Query"):
            st.write("Data represents GDP_contribution in %")
        hf2=pd.read_csv("upcoming_sectors.csv")
        

        fig = px.scatter(
            hf2,
            x="Sector",
            y="GDP_Contribution_%",
            size="GDP_Contribution_%",
            color="Specification",
            hover_name="Sector",
            size_max=90,               
            width=1600,              
            height=500,                
            title="Upcoming Sectors  GDP Contribution Bubble Chart",
        )

        
        fig.update_layout(
            xaxis_title="Sectors",
            yaxis_title="GDP Contribution (%)",
            font=dict(size=18),
        )
        st.plotly_chart(fig, use_container_width=True)
        ####################################
    with tab3:
        st.success("Our Responsibilty to give you Best Analysis/Prediction..")
        sf1=pd.read_csv("current_sectors.csv")
        sf2=pd.read_csv("upcoming_sectors.csv")
        if selected_option=="Student":
            st.caption("WE will give you future and currently growing career opportunities for your growth and you can analyze and find out which is better for you")
            selected1_option = st.selectbox(
            label="Experience in your field",
            options=["Fresher", "1-3 years", ">3 years"] ,index=None,placeholder="Your experience-")
            st.write(f"You selected: {selected1_option}")
            if selected1_option=="Fresher":
                st.info("The best sectors in which you can dive are..")
                st.header("From Current Sectors*")
                top3 = sf1.sort_values(by="GDP_Contribution_%", ascending=False).head(3)
                st.dataframe(top3)
                st.header("From Upcoming Sectors*")
                top3a = sf2.sort_values(by="GDP_Contribution_%", ascending=False).head(3)
                st.dataframe(top3a)
            elif selected1_option=="1-3 years":
                st.info("The best sectors in which you dive are ,where you have sufficient knowledge already.")
                st.header("From Current Sectors*")
                top7 = sf1.sort_values(by="GDP_Contribution_%", ascending=False).head(7)
                st.dataframe(top7)
                st.header("From Upcoming Sectors*")
                top7a = sf2.sort_values(by="GDP_Contribution_%", ascending=False).head(7)
                st.dataframe(top7a)
            elif selected1_option==">3 years":
                st.info("Now you have great experience in at most sectors ,So  now you can focus on Emerging and Highly growing sectors..")  
                st.header("From Current Sectors*")
                filtered = sf1[
                sf1["Specification"].str.contains("Emerging|Highly Growing", case=False, na=False)]
                st.table(filtered)
                st.header("From Upcoming Sectors*")
                filtered1 = sf2[
                sf2["Specification"].str.contains("Emerging|Highly Growing", case=False, na=False)]
                st.table(filtered1)
            else:
                pass  
            st.divider()
            st.subheader("--Job Opportunities in various fields..")
 

            data = {
                "Sector": [
                    "IT Services",
                    "Healthcare & Life Sciences",
                    "Education",
                    "Food & Beverages",
                    "Construction",
                    "Human Resources (HR Services)",
                    "Agriculture / Agritech",
                    "FinTech / Financial Technology"
                ],
                "Direct_Jobs_Created": [
                    204119,
                    147639,
                    90414,
                    88468,
                    88702,
                    87983,
                    83307,
                    56819
                ]
            }

            df = pd.DataFrame(data)

            
            fig = px.area(
                df,
                x="Sector",
                y="Direct_Jobs_Created",
                title="Direct Jobs Created by Sector",
                markers=True
            )

            st.plotly_chart(fig, use_container_width=True)
            st.divider()
  
        elif selected_option=="Employee":
            st.caption("WE will give some additional features for analysis so you can add molre values in an existing career or future shift..")
            st.write("**This are the sectors beneficial for you ,after analysis and market prediction this are the suggestions**")
            with st.expander("You Opportunities in the following Sectors.."):
                ee1=["Artificial Intelligence (AI)","Smart Manufacturing (Industry 4.0)","Green Hydrogen / Renewable Energy / EV Ecosystem (Energy Transition)"]
                ee2=["IT & ITES (IT-BPM)","Financial Services (FinTech + Banking)","Pharmaceuticals / Healthcare"]
                st.header("--Current sectors--")
                st.table(ee1)
                st.header("--Upcoming sectors--")
                st.table(ee2)
            with st.expander("Let's Visualize"):
                
                st.title("Sector Analysis: GDP vs Employment")


                dfn= pd.read_csv("research11.csv")

                fig = px.scatter(
                    dfn,
                    x='Sector',
                    y='GDP_Share_%',
                    size='Employment_Share_%',
                    color='Growth_Trend',
                    hover_data=['GDP_Share_%', 'Employment_Share_%'],  # Hover info
                    size_max=40,
                    title="Sector GDP vs Employment with Growth Trend"
                )

                fig.update_layout(xaxis_tickangle=-45, yaxis_title="GDP Share %", xaxis_title="Sector")


                st.plotly_chart(fig, use_container_width=True)
                
            
        elif selected_option=="Entrepreneur":
            st.caption("WE will give analysis according to market trends and unique analysis for your efficient or drastic growth..")
            tab01,tab02,tab03,tab04=st.tabs(["Stacked Area Chart","Tabular data","Bar Chart","Line chart"])
            with tab01:
                kk1 = pd.read_csv("entrepreneur11.csv")

                sectors = ["IT_Services","Healthcare","Education","Professional_Services",
                        "Agriculture","Food_and_Beverages","FinTech","Others"]

                kk1["Year"] = pd.to_numeric(kk1["Year"], errors="coerce")
                for col in sectors:
                    kk1[col] = pd.to_numeric(kk1[col], errors="coerce")

                area_df = kk1.melt(id_vars="Year", value_vars=sectors,
                                var_name="Sector", value_name="Startups")

                st.title("Startup Numbers by Sector (2016–2024) – Colored Area Chart")


                custom_colors = [
                    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
                ]

                chart = alt.Chart(area_df).mark_area(opacity=0.7).encode(
                    x=alt.X("Year:O", title="Year"),
                    y=alt.Y("Startups:Q", title="Number of Startups"),
                    color=alt.Color("Sector:N", scale=alt.Scale(range=custom_colors)),
                    tooltip=["Year", "Sector", "Startups"]
                ).properties(
                    width=900,
                    height=500
                )

                st.altair_chart(chart, use_container_width=True)

            with tab02:
                st.table(kk1) 
            with tab03:
                kk1 = pd.read_csv("entrepreneur11.csv")

                sectors = ["IT_Services","Healthcare","Education","Professional_Services",
                        "Agriculture","Food_and_Beverages","FinTech","Others"]

                kk1["Year"] = pd.to_numeric(kk1["Year"], errors="coerce")
                for col in sectors:
                    kk1[col] = pd.to_numeric(kk1[col], errors="coerce")
                long_df = kk1.melt(id_vars="Year", value_vars=sectors,
                                var_name="Sector", value_name="Startups")

                st.bar_chart(long_df, x="Year", y="Startups", color="Sector")
            with tab04:
                kk1 = pd.read_csv("entrepreneur11.csv")
                data = kk1[["Year", "Total_Startups"]]
                st.line_chart(data, x="Year", y="Total_Startups")



        elif selected_option=="Researcher":
            st.caption("We will give analysis and prediction visualization to the next level..") 
            tabs91,tabs93,tabs94=st.tabs(["Terms Info.",":material/all_inclusive:","Formualaes for Calculation"])
            with tabs91:
                with st.expander("Just Rhythming Casual Concepts.."):
                    st.write("In India, the most frequently used terms similar to and related to Gross Domestic Product (GDP) are its component and derivative measures such as Gross Value Added (GVA), Gross National Product (GNP), and Net Domestic Product (NDP)")
                with st.expander("Key GDP like terms.."):
                    data = {
                    "Term": [
                        "GVA (Gross Value Added)",
                        "GNP (Gross National Product)",
                        "GNI (Gross National Income)",
                        "NDP (Net Domestic Product)",
                        "NNP (Net National Product)",
                        "Per Capita Income",
                        "Nominal GDP",
                        "Real GDP",
                        "GDP per Capita",
                        "PPP (Purchasing Power Parity)",
                        "Inflation",
                        "CPI (Consumer Price Index)",
                        "WPI (Wholesale Price Index)",
                        "Fiscal Deficit"
                    ],
                    "Meaning": [
                        "Measures value added by a sector by subtracting intermediate inputs from output.",
                        "Total value of goods/services produced by Indian residents, including income abroad.",
                        "Modern term for GNP; total income of residents from domestic and foreign sources.",
                        "GDP minus depreciation, showing net domestic production.",
                        "GNP minus depreciation to measure national net income.",
                        "National income divided by population to show average income.",
                        "GDP measured at current prices without inflation adjustment.",
                        "GDP adjusted for inflation to reflect real output change.",
                        "GDP divided by population to indicate average living standards.",
                        "GDP adjusted for price differences between countries for comparison.",
                        "Rate at which general prices rise, reducing purchasing power.",
                        "Measures retail price changes of household goods.",
                        "Measures price changes of goods at the wholesale level.",
                        "Gap between government spending and non-borrowed income, showing borrowing need."
                    ]
                }

                    
                    st.table(data)
                with st.expander("GDP related Economic Indicators.."):
                    

                    data = {
                        "Term": [
                            "Nominal GDP",
                            "Real GDP",
                            "GDP per Capita",
                            "Purchasing Power Parity (PPP)",
                            "Inflation",
                            "Consumer Price Index (CPI)",
                            "Wholesale Price Index (WPI)",
                            "Fiscal Deficit"
                        ],
                        "Meaning": [
                            "GDP measured at current market prices without adjusting for inflation.",
                            "GDP adjusted for inflation to reflect actual change in output.",
                            "GDP divided by population to indicate average living standards.",
                            "GDP adjusted for differences in price levels across countries to compare economic size realistically.",
                            "The rate at which the general level of prices rises over time, reducing purchasing power.",
                            "Measures the change in retail prices of a basket of goods consumed by households.",
                            "Measures changes in prices of goods at the wholesale level before retail sale.",
                            "The gap between government expenditure and non-borrowed revenue, indicating how much the government must borrow."
                        ]
                    }

                    st.table(data)
   
            with tabs93:
                st.success("Linked") 
            with tabs94:
                selected_formula=st.selectbox("Choose any given Formula..",["GDP","GVA","GNP","NDP"],index=None,placeholder="Formula for predicting..")  
                if selected_formula=="GDP":
                    st.info("GDP=Consumption Expenditure+Investment Expenditure+Government Spending+(X-M)") 
                    st.info("(X-M)==>Net Exports--->(Exports-Imports)")
                    st.divider()
                    st.info("All Values in Lakh Crore..")
                    st.divider()
                    C= st.number_input("Enter the Consumption:", value=0.0, step=0.1, format="%.2f")
                    I=st.number_input("Enter the Investment:", value=0.0, step=0.1, format="%.2f")
                    G=st.number_input("Enter the Gov. spending:", value=0.0, step=0.1, format="%.2f")
                    XM=st.number_input("Enter the Net EXports:", value=0.0, step=0.1, format="%.2f")
                    gdp=C+I+G+XM
                    st.success(f"The Calculated GDP is :{gdp}%")
                    st.divider() 
                elif selected_formula=="GVA":
                    st.info("GVA = GDP + Subsidies on Products – Taxes on Products")
                    st.divider()
                    st.info("All Values in Lakhs..")
                    st.divider()
                    gp= st.number_input("Enter the GDP:", value=0.0, step=0.1, format="%.2f") 
                    ss= st.number_input("Enter the Subsidies:", value=0.0, step=0.1, format="%.2f")
                    ts= st.number_input("Enter the Taxes:", value=0.0, step=0.1, format="%.2f")
                    gva=gp+ss-ts
                    st.success(f"The Calculated GVA is:{gva}%")
                    st.divider()
                elif selected_formula=="GNP":
                    st.info("GNP = GDP + Net Factor Income from Abroad")  
                    st.divider()
                    st.info("All Values in Lakh Crore..")
                    st.divider()
                    gp= st.number_input("Enter the GDP:", value=0.0, step=0.1, format="%.2f") 
                    nf= st.number_input("Enter the Net factor:", value=0.0, step=0.1, format="%.2f")
                    gnp=gp+nf
                    st.success(f"The Calculate GNP is:{gnp}%")
                    st.divider()
                elif selected_formula=="NDP":
                    st.info("NDP = GDP - Depreciation") 
                    st.divider()
                    st.info("All Values in Lakh Crore..")
                    st.divider()
                    gpp= st.number_input("Enter the GDP:", value=0.0, step=0.1, format="%.2f") 
                    dp= st.number_input("Enter the Depreciation:", value=0.0, step=0.1, format="%.2f")
                    ndp=gpp-dp
                    st.success(f"The Calculated NDP is :{ndp}%")
                    st.divider()
                else:
                    pass    






                    


        else:
            pass
              
                    
            
