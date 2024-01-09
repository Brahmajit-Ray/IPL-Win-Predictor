import streamlit as st
import pickle
import pandas as pd

st.title("IPL Win Predictor")

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Punjab Kings',
 'Chennai Super Kings']

cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah','Mohali', 'Bengaluru']

col1,col2= st.columns(2)

model=pickle.load(open("model1.pkl","rb"))

with col1:
    batting_team=st.selectbox("Select the batting team",sorted(teams))
with col2:
    bowling_team=st.selectbox("Select the bowling team",sorted(teams))

selected_city=st.selectbox("Select venue",sorted(cities))

target=st.number_input("Target",min_value=0,step=1)

col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input("Score",min_value=0,step=1)
with col4:
    overs=st.number_input("Overs",min_value=1,max_value=19,step=1)
with col5:
    wickets=st.number_input("Wickets",min_value=0,max_value=9,step=1)

if st.button("Predict Probability"):
    runs_left=target-score
    balls_left=120-overs*6
    wickets_left=10-wickets
    crr=score/overs
    rrr=runs_left*6/balls_left

    df=pd.DataFrame({"batting_team":[batting_team],"bowling_team":[bowling_team],
                     "city":[selected_city],"runs_left":[runs_left],"balls_left":[balls_left],
                     "wickets_left":[wickets_left],"total_runs_x":[target],"crr":[crr],
                     "rrr":[rrr]})



    res=model.predict_proba(df)
    win1=res[0][0]
    win2=res[0][1]

    st.header(batting_team+"-"+str(round(win2*100,1))+"%")
    st.header(bowling_team+"-"+str(round(win1*100,1))+"%")



