import pandas as pd
import streamlit as st
from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import style
from src.match_tab import display_match_tab

# Apply custom style
style.apply_custom_styles()

about_tab,match_tab, player_tab, prediction_tab = st.tabs(["About IPL","Team performance", "Players performance","prediction"])


# Load data
@st.cache_data
def load_data():
    t = pd.read_csv('data/teams.csv')
    p = pd.read_csv('data/players.csv')
    m = pd.read_csv('data/matches.csv')
    d = pd.read_csv('data/deliveries.csv')
    return t, p, m, d


teams, players, matches, deliveries = load_data()

# Sidebar
with st.sidebar:
    col1, mid, col2 = st.columns([1, 3, 20])
    with col1:
        st.image('images/logo.png', width=55)
    with col2:
        st.title(':rainbow[Insights]')

    team_list = teams['team'].unique()
    selected_team = st.selectbox(':blue[Select a team]', team_list)
    year_list = teams.loc[teams['team'] == selected_team, 'year'].unique()[::-1]
    selected_years = st.multiselect(
        ':blue[Select Seasons]',
        year_list,
        year_list[:6])

display_match_tab(match_tab, matches, selected_team, selected_years)


with about_tab:
    st.title("About IPL Prediction Engine and Analysis")
    st.markdown(
        """
        This IPL Prediction Engine and analysis application is designed to provide match predictions for IPL matches based on historical match data and machine learning models. It utilizes detailed match-by-match data for IPL matches available on [https://www.iplt20.com/] to train the ML model going back to 2008 and predicts match outcomes for the 2021 season.

        ### Methodology
        The model uses various features based on a team's historical and recent performances to predict match outcomes. It is re-run after every single match in the tournament to update win probabilities for remaining games.

        ### Data Sources: [https://www.iplt20.com/]

        ### About
        This application is developed by Hitesh Salimath 2347118. 
        It is intended for educational and informational purposes only.
        """
    )


with player_tab:
    st.title("Development in progress...")
    detail=pd.read_csv('IPL Ball-by-Ball 2008-2020.csv')
    match=pd.read_csv('IPL Matches 2008-2020.csv')
    match['season']=pd.DatetimeIndex(match['date']).year
    new=detail.merge(match,left_on='id',right_on='id')

    batsman=['A Ashish Reddy', 'A Chandila', 'A Chopra', 'A Choudhary',
        'A Dananjaya', 'A Flintoff', 'A Kumble', 'A Mishra', 'A Mithun',
        'A Mukund', 'A Nehra', 'A Nortje', 'A Singh', 'A Symonds',
        'A Uniyal', 'A Zampa', 'AA Bilakhia', 'AA Chavan',
        'AA Jhunjhunwala', 'AA Noffke', 'AB Agarkar', 'AB Barath',
        'AB Dinda', 'AB McDonald', 'AB de Villiers', 'AC Blizzard',
        'AC Gilchrist', 'AC Thomas', 'AC Voges', 'AD Hales',
        'AD Mascarenhas', 'AD Mathews', 'AD Nath', 'AD Russell',
        'AF Milne', 'AG Murtaza', 'AG Paunikar', 'AJ Finch', 'AJ Turner',
        'AJ Tye', 'AL Menaria', 'AM Nayar', 'AM Rahane', 'AN Ahmed',
        'AN Ghosh', 'AP Dole', 'AP Majumdar', 'AP Tare', 'AR Bawne',
        'AR Patel', 'AS Joseph', 'AS Rajpoot', 'AS Raut', 'AS Yadav',
        'AT Carey', 'AT Rayudu', 'AUK Pathan', 'Abdul Samad',
        'Abdur Razzak', 'Abhishek Sharma', 'Anirudh Singh', 'Ankit Sharma',
        'Ankit Soni', 'Anureet Singh', 'Arshdeep Singh', 'Avesh Khan',
        'Azhar Mahmood', 'B Akhil', 'B Chipli', 'B Kumar', 'B Laughlin',
        'B Lee', 'B Stanlake', 'B Sumanth', 'BA Bhatt', 'BA Stokes',
        'BAW Mendis', 'BB McCullum', 'BB Samantray', 'BB Sran',
        'BCJ Cutting', 'BE Hendricks', 'BJ Haddin', 'BJ Hodge',
        'BJ Rohrer', 'BMAJ Mendis', 'BR Dunk', 'Basil Thampi',
        'Bipul Sharma', 'C Madan', 'C Munro', 'C Nanda', 'C de Grandhomme',
        'CA Ingram', 'CA Lynn', 'CA Pujara', 'CH Gayle', 'CH Morris',
        'CJ Anderson', 'CJ Ferguson', 'CJ Jordan', 'CJ McKay',
        'CK Kapugedera', 'CK Langeveldt', 'CL White', 'CM Gautam',
        'CR Brathwaite', 'CR Woakes', 'CRD Fernando', 'CV Varun',
        'D Kalyankrishna', 'D Padikkal', 'D Salunkhe', 'D Wiese',
        'D du Preez', 'DA Miller', 'DA Warner', 'DAJ Bracewell', 'DB Das',
        'DB Ravi Teja', 'DE Bollinger', 'DH Yagnik', 'DJ Bravo',
        'DJ Harris', 'DJ Hooda', 'DJ Hussey', 'DJ Jacobs', 'DJ Muthuswami',
        'DJ Thornely', 'DJG Sammy', 'DJM Short', 'DL Chahar', 'DL Vettori',
        'DM Bravo', 'DNT Zoysa', 'DP Nannes', 'DP Vijaykumar',
        'DPMD Jayawardene', 'DR Martyn', 'DR Sams', 'DR Shorey',
        'DR Smith', 'DS Kulkarni', 'DS Lehmann', 'DT Christian',
        'DT Patil', 'DW Steyn', 'E Lewis', 'EJG Morgan', 'ER Dwivedi',
        'F Behardien', 'F du Plessis', 'FH Edwards', 'FY Fazal',
        'G Gambhir', 'GB Hogg', 'GC Smith', 'GC Viljoen', 'GD McGrath',
        'GH Vihari', 'GJ Bailey', 'GJ Maxwell', 'GR Napier',
        'Gurkeerat Singh', 'H Das', 'H Klaasen', 'HF Gurney', 'HH Gibbs',
        'HH Pandya', 'HM Amla', 'HV Patel', 'Harbhajan Singh',
        'Harmeet Singh', 'Harpreet Brar', 'Harpreet Singh', 'I Malhotra',
        'I Sharma', 'I Udana', 'IC Pandey', 'IK Pathan', 'IR Jaggi',
        'IS Sodhi', 'Imran Tahir', 'Iqbal Abdulla', 'Ishan Kishan',
        'J Arunkumar', 'J Botha', 'J Suchith', 'J Syed Mohammad',
        'J Theron', 'J Yadav', 'JA Morkel', 'JC Archer', 'JC Buttler',
        'JD Ryder', 'JD Unadkat', 'JDP Oram', 'JDS Neesham', 'JE Taylor',
        'JEC Franklin', 'JH Kallis', 'JJ Bumrah', 'JJ Roy',
        'JJ van der Wath', 'JL Denly', 'JL Pattinson', 'JM Bairstow',
        'JM Kemp', 'JO Holder', 'JP Duminy', 'JP Faulkner',
        'JPR Scantlebury-Searles', 'JR Hopes', 'JR Philippe',
        'Jaskaran Singh', 'Joginder Sharma', 'K Goel', 'K Gowtham',
        'K Rabada', 'K Upadhyay', 'KA Pollard', 'KAJ Roach',
        'KB Arun Karthik', 'KC Cariappa', 'KC Sangakkara', 'KD Karthik',
        'KH Pandya', 'KJ Abbott', 'KK Ahmed', 'KK Cooper', 'KK Nair',
        'KL Nagarkoti', 'KL Rahul', 'KM Jadhav', 'KMA Paul',
        'KMDN Kulasekara', 'KP Appanna', 'KP Pietersen', 'KS Williamson',
        'KV Sharma', 'KW Richardson', 'Kamran Akmal', 'Kamran Khan',
        'Karanveer Singh', 'Kartik Tyagi', 'Kuldeep Yadav', 'L Ablish',
        'L Balaji', 'L Ronchi', 'LA Carseldine', 'LA Pomersbach',
        'LE Plunkett', 'LH Ferguson', 'LJ Wright', 'LMP Simmons',
        'LPC Silva', 'LR Shukla', 'LRPL Taylor', 'LS Livingstone',
        'M Ashwin', 'M Kaif', 'M Kartik', 'M Klinger', 'M Manhas',
        'M Markande', 'M Morkel', 'M Muralitharan', 'M Ntini',
        'M Prasidh Krishna', 'M Rawat', 'M Vijay', 'M Vohra', 'M de Lange',
        'MA Agarwal', 'MA Khote', 'MA Starc', 'MA Wood', 'MC Henriques',
        'MC Juneja', 'MD Mishra', 'MDKJ Perera', 'MEK Hussey',
        'MF Maharoof', 'MG Johnson', 'MJ Clarke', 'MJ Guptill', 'MJ Lumb',
        'MJ McClenaghan', 'MJ Santner', 'MK Lomror', 'MK Pandey',
        'MK Tiwary', 'ML Hayden', 'MM Ali', 'MM Patel', 'MM Sharma',
        'MN Samuels', 'MN van Wyk', 'MP Stoinis', 'MR Marsh', 'MS Bisla',
        'MS Dhoni', 'MS Gony', 'MS Wade', 'MV Boucher', 'Mandeep Singh',
        'Mashrafe Mortaza', 'Misbah-ul-Haq', 'Mohammad Ashraful',
        'Mohammad Asif', 'Mohammad Hafeez', 'Mohammad Nabi',
        'Mohammed Shami', 'Mohammed Siraj', 'Mujeeb Ur Rahman',
        'Mustafizur Rahman', 'N Jagadeesan', 'N Pooran', 'N Rana',
        'N Saini', 'NA Saini', 'ND Doshi', 'NJ Maddinson', 'NJ Rimmington',
        'NL McCullum', 'NLTC Perera', 'NM Coulter-Nile', 'NS Naik',
        'NV Ojha', 'Niraj Patel', 'OA Shah', 'P Awana', 'P Chopra',
        'P Dogra', 'P Dubey', 'P Kumar', 'P Negi', 'P Parameswaran',
        'P Ray Barman', 'P Sahu', 'P Simran Singh', 'PA Patel', 'PA Reddy',
        'PC Valthaty', 'PD Collingwood', 'PJ Cummins', 'PJ Sangwan',
        'PK Garg', 'PM Sarvesh Kumar', 'PP Chawla', 'PP Ojha', 'PP Shaw',
        'PR Shah', 'PSP Handscomb', 'PV Tambe', 'Pankaj Singh',
        'Parvez Rasool', 'Q de Kock', 'R Ashwin', 'R Bhatia', 'R Bishnoi',
        'R Dhawan', 'R Dravid', 'R McLaren', 'R Ninan', 'R Parag',
        'R Rampaul', 'R Sathish', 'R Sharma', 'R Shukla', 'R Tewatia',
        'R Vinay Kumar', 'RA Jadeja', 'RA Tripathi', 'RD Chahar',
        'RD Gaikwad', 'RE Levi', 'RE van der Merwe', 'RG More',
        'RG Sharma', 'RJ Harris', 'RJ Peterson', 'RJ Quiney', 'RK Bhui',
        'RK Singh', 'RN ten Doeschate', 'RP Singh', 'RR Bhatkal',
        'RR Pant', 'RR Powar', 'RR Raje', 'RR Rossouw', 'RR Sarwan',
        'RS Bopara', 'RS Gavaskar', 'RS Sodhi', 'RT Ponting', 'RV Gomez',
        'RV Uthappa', 'Rashid Khan', 'Rasikh Salam', 'Ravi Bishnoi',
        'S Anirudha', 'S Aravind', 'S Badree', 'S Badrinath',
        'S Chanderpaul', 'S Dhawan', 'S Dube', 'S Gopal', 'S Kaul',
        'S Kaushik', 'S Ladda', 'S Lamichhane', 'S Nadeem', 'S Narwal',
        'S Rana', 'S Randiv', 'S Sohal', 'S Sreesanth', 'S Sriram',
        'S Tyagi', 'S Vidyut', 'SA Abbott', 'SA Asnodkar', 'SA Yadav',
        'SB Bangar', 'SB Jakati', 'SB Joshi', 'SB Styris', 'SB Wagh',
        'SC Ganguly', 'SD Chitnis', 'SD Lad', 'SE Bond', 'SE Marsh',
        'SE Rutherford', 'SJ Srivastava', 'SK Raina', 'SK Trivedi',
        'SK Warne', 'SL Malinga', 'SM Curran', 'SM Harwood', 'SM Katich',
        'SM Pollock', 'SMSM Senanayake', 'SN Khan', 'SN Thakur',
        'SO Hetmyer', 'SP Fleming', 'SP Goswami', 'SP Jackson',
        'SP Narine', 'SPD Smith', 'SR Tendulkar', 'SR Watson',
        'SS Cottrell', 'SS Iyer', 'SS Shaikh', 'SS Tiwary',
        'ST Jayasuriya', 'STR Binny', 'SV Samson', 'SW Billings',
        'SW Tait', 'Sachin Baby', 'Salman Butt', 'Sandeep Sharma',
        'Shahbaz Ahmed', 'Shahid Afridi', 'Shakib Al Hasan', 'Shivam Mavi',
        'Shivam Sharma', 'Shoaib Ahmed', 'Shoaib Akhtar', 'Shoaib Malik',
        'Shubman Gill', 'Sohail Tanvir', 'Sunny Gupta', 'Sunny Singh',
        'Swapnil Singh', 'T Banton', 'T Henderson', 'T Kohli',
        'T Natarajan', 'T Taibu', 'T Thushara', 'TA Boult', 'TD Paine',
        'TG Southee', 'TK Curran', 'TL Suman', 'TM Dilshan', 'TM Head',
        'TM Srivastava', 'TR Birt', 'TS Mills', 'TU Deshpande', 'U Kaul',
        'UA Birla', 'UBT Chand', 'UT Khawaja', 'UT Yadav', 'Umar Gul',
        'V Kohli', 'V Pratap Singh', 'V Sehwag', 'V Shankar', 'VH Zol',
        'VR Aaron', 'VRV Singh', 'VS Malik', 'VS Yeligati', 'VVS Laxman',
        'VY Mahesh', 'Vishnu Vinod', 'W Jaffer', 'WA Mota', 'WD Parnell',
        'WP Saha', 'WPUJC Vaas', 'Washington Sundar',
        'X Thalaivan Sargunam', 'Y Gnaneswara Rao', 'Y Nagar',
        'Y Prithvi Raj', 'Y Venugopal Rao', 'YA Abdulla', 'YBK Jaiswal',
        'YK Pathan', 'YS Chahal', 'YV Takawale', 'Yashpal Singh',
        'Younis Khan', 'Yuvraj Singh', 'Z Khan']

    bowler=['A Ashish Reddy', 'A Chandila', 'A Choudhary', 'A Dananjaya',
        'A Flintoff', 'A Kumble', 'A Mishra', 'A Mithun', 'A Nehra',
        'A Nel', 'A Nortje', 'A Singh', 'A Symonds', 'A Uniyal', 'A Zampa',
        'AA Chavan', 'AA Jhunjhunwala', 'AA Kazi', 'AA Noffke',
        'AB Agarkar', 'AB Dinda', 'AB McDonald', 'AC Gilchrist',
        'AC Thomas', 'AC Voges', 'AD Mascarenhas', 'AD Mathews',
        'AD Russell', 'AF Milne', 'AG Murtaza', 'AJ Finch', 'AJ Tye',
        'AL Menaria', 'AM Nayar', 'AM Rahane', 'AM Salvi', 'AN Ahmed',
        'AP Dole', 'AR Patel', 'AS Joseph', 'AS Rajpoot', 'AS Raut',
        'AS Roy', 'AUK Pathan', 'Abdul Samad', 'Abdur Razzak',
        'Abhishek Sharma', 'Anand Rajan', 'Ankit Sharma', 'Ankit Soni',
        'Anureet Singh', 'Arshdeep Singh', 'Avesh Khan', 'Azhar Mahmood',
        'B Akhil', 'B Chipli', 'B Geeves', 'B Kumar', 'B Laughlin',
        'B Lee', 'B Stanlake', 'BA Bhatt', 'BA Stokes', 'BAW Mendis',
        'BB Sran', 'BCJ Cutting', 'BE Hendricks', 'BJ Hodge', 'BJ Rohrer',
        'BMAJ Mendis', 'BW Hilfenhaus', 'Basil Thampi', 'Bipul Sharma',
        'C Ganapathy', 'C Munro', 'C Nanda', 'C de Grandhomme', 'CH Gayle',
        'CH Morris', 'CJ Anderson', 'CJ Dala', 'CJ Green', 'CJ Jordan',
        'CJ McKay', 'CK Kapugedera', 'CK Langeveldt', 'CL White',
        'CR Brathwaite', 'CR Woakes', 'CRD Fernando', 'CV Varun',
        'D Kalyankrishna', 'D Salunkhe', 'D Wiese', 'D du Preez',
        'DA Warner', 'DAJ Bracewell', 'DB Ravi Teja', 'DE Bollinger',
        'DJ Bravo', 'DJ Harris', 'DJ Hooda', 'DJ Hussey', 'DJ Muthuswami',
        'DJ Thornely', 'DJ Willey', 'DJG Sammy', 'DJM Short', 'DL Chahar',
        'DL Vettori', 'DNT Zoysa', 'DP Nannes', 'DP Vijaykumar', 'DR Sams',
        'DR Smith', 'DS Kulkarni', 'DT Christian', 'DW Steyn',
        'F du Plessis', 'FH Edwards', 'FY Fazal', 'GB Hogg', 'GC Viljoen',
        'GD McGrath', 'GH Vihari', 'GJ Maxwell', 'GR Napier', 'GS Sandhu',
        'Gagandeep Singh', 'Gurkeerat Singh', 'HF Gurney', 'HH Pandya',
        'HV Patel', 'Harbhajan Singh', 'Harmeet Singh',
        'Harmeet Singh (2)', 'Harpreet Brar', 'I Malhotra', 'I Sharma',
        'I Udana', 'IC Pandey', 'IK Pathan', 'IS Sodhi', 'Imran Tahir',
        'Iqbal Abdulla', 'J Botha', 'J Suchith', 'J Syed Mohammad',
        'J Theron', 'J Yadav', 'JA Morkel', 'JC Archer', 'JD Ryder',
        'JD Unadkat', 'JDP Oram', 'JDS Neesham', 'JE Taylor',
        'JEC Franklin', 'JH Kallis', 'JJ Bumrah', 'JJ van der Wath',
        'JL Pattinson', 'JM Kemp', 'JO Holder', 'JP Behrendorff',
        'JP Duminy', 'JP Faulkner', 'JPR Scantlebury-Searles',
        'JR Hazlewood', 'JR Hopes', 'JW Hastings', 'Jaskaran Singh',
        'Joginder Sharma', 'K Goel', 'K Gowtham', 'K Khejroliya',
        'K Rabada', 'K Santokie', 'K Upadhyay', 'KA Pollard', 'KAJ Roach',
        'KC Cariappa', 'KH Pandya', 'KJ Abbott', 'KK Ahmed', 'KK Cooper',
        'KL Nagarkoti', 'KM Asif', 'KMA Paul', 'KMDN Kulasekara',
        'KP Appanna', 'KP Pietersen', 'KS Williamson', 'KV Sharma',
        'KW Richardson', 'Kamran Khan', 'Karanveer Singh', 'Kartik Tyagi',
        'Kuldeep Yadav', 'L Ablish', 'L Balaji', 'L Ngidi',
        'LA Carseldine', 'LE Plunkett', 'LH Ferguson', 'LJ Wright',
        'LMP Simmons', 'LPC Silva', 'LR Shukla', 'LRPL Taylor',
        'LS Livingstone', 'M Ashwin', 'M Kartik', 'M Manhas', 'M Markande',
        'M Morkel', 'M Muralitharan', 'M Ntini', 'M Prasidh Krishna',
        'M Vijay', 'M de Lange', 'MA Khote', 'MA Starc', 'MA Wood',
        'MB Parmar', 'MC Henriques', 'MF Maharoof', 'MG Johnson',
        'MG Neser', 'MJ Clarke', 'MJ Henry', 'MJ McClenaghan',
        'MJ Santner', 'MK Lomror', 'MK Tiwary', 'MM Ali', 'MM Patel',
        'MM Sharma', 'MN Samuels', 'MP Stoinis', 'MR Marsh', 'MS Gony',
        'Mandeep Singh', 'Mashrafe Mortaza', 'Mohammad Asif',
        'Mohammad Hafeez', 'Mohammad Nabi', 'Mohammed Shami',
        'Mohammed Siraj', 'Monu Kumar', 'Mujeeb Ur Rahman',
        'Mustafizur Rahman', 'N Rana', 'NA Saini', 'NB Singh', 'ND Doshi',
        'NJ Rimmington', 'NL McCullum', 'NLTC Perera', 'NM Coulter-Nile',
        'O Thomas', 'P Amarnath', 'P Awana', 'P Dubey', 'P Kumar',
        'P Negi', 'P Parameswaran', 'P Prasanth', 'P Ray Barman', 'P Sahu',
        'P Suyal', 'PC Valthaty', 'PD Collingwood', 'PJ Cummins',
        'PJ Sangwan', 'PM Sarvesh Kumar', 'PP Chawla', 'PP Ojha',
        'PV Tambe', 'Pankaj Singh', 'Parvez Rasool', 'R Ashwin',
        'R Bhatia', 'R Dhawan', 'R McLaren', 'R Ninan', 'R Parag',
        'R Rampaul', 'R Sathish', 'R Sharma', 'R Shukla', 'R Tewatia',
        'R Vinay Kumar', 'RA Jadeja', 'RA Shaikh', 'RA Tripathi',
        'RD Chahar', 'RE van der Merwe', 'RG More', 'RG Sharma',
        'RJ Harris', 'RJ Peterson', 'RN ten Doeschate', 'RP Singh',
        'RR Bhatkal', 'RR Bose', 'RR Powar', 'RR Raje', 'RS Bopara',
        'RS Gavaskar', 'RV Gomez', 'RW Price', 'Rashid Khan',
        'Rasikh Salam', 'Ravi Bishnoi', 'S Aravind', 'S Badree',
        'S Dhawan', 'S Dube', 'S Gopal', 'S Kaul', 'S Kaushik', 'S Ladda',
        'S Lamichhane', 'S Midhun', 'S Nadeem', 'S Narwal', 'S Rana',
        'S Randiv', 'S Sandeep Warrier', 'S Sreesanth', 'S Sriram',
        'S Tyagi', 'S Vidyut', 'SA Abbott', 'SA Yadav', 'SB Bangar',
        'SB Jakati', 'SB Joshi', 'SB Styris', 'SB Wagh', 'SC Ganguly',
        'SC Kuggeleijn', 'SD Chitnis', 'SE Bond', 'SE Rutherford',
        'SJ Srivastava', 'SK Raina', 'SK Trivedi', 'SK Warne',
        'SL Malinga', 'SM Boland', 'SM Curran', 'SM Harwood', 'SM Pollock',
        'SMSM Senanayake', 'SN Khan', 'SN Thakur', 'SP Narine',
        'SPD Smith', 'SR Tendulkar', 'SR Watson', 'SS Agarwal',
        'SS Cottrell', 'SS Mundhe', 'SS Sarkar', 'ST Jayasuriya',
        'STR Binny', 'SW Tait', 'Sachin Baby', 'Sandeep Sharma',
        'Shahbaz Ahmed', 'Shahid Afridi', 'Shakib Al Hasan', 'Shivam Mavi',
        'Shivam Sharma', 'Shoaib Ahmed', 'Shoaib Akhtar', 'Shoaib Malik',
        'Sohail Tanvir', 'Sunny Gupta', 'Swapnil Singh', 'T Henderson',
        'T Natarajan', 'T Shamsi', 'T Thushara', 'TA Boult', 'TG Southee',
        'TK Curran', 'TL Suman', 'TM Dilshan', 'TM Head', 'TP Sudhindra',
        'TS Mills', 'TU Deshpande', 'Tejas Baroka', 'UT Yadav', 'Umar Gul',
        'V Kohli', 'V Pratap Singh', 'V Sehwag', 'V Shankar', 'VR Aaron',
        'VRV Singh', 'VS Malik', 'VS Yeligati', 'VY Mahesh', 'WA Mota',
        'WD Parnell', 'WPUJC Vaas', 'Washington Sundar',
        'Y Gnaneswara Rao', 'Y Nagar', 'Y Prithvi Raj', 'Y Venugopal Rao',
        'YA Abdulla', 'YK Pathan', 'YS Chahal', 'Yuvraj Singh', 'Z Khan']

    death_players=('A Ashish Reddy',
    'A Mishra',
    'A Nehra',
    'A Symonds',
    'AB Agarkar',
    'AB McDonald',
    'AB de Villiers',
    'AD Mathews',
    'AD Russell',
    'AJ Finch',
    'AJ Tye',
    'AL Menaria',
    'AM Nayar',
    'AM Rahane',
    'AR Patel',
    'AS Raut',
    'AT Rayudu',
    'Abhishek Sharma',
    'Azhar Mahmood',
    'B Kumar',
    'B Lee',
    'BA Stokes',
    'BB McCullum',
    'BCJ Cutting',
    'BJ Hodge',
    'Bipul Sharma',
    'C de Grandhomme',
    'CH Gayle',
    'CH Morris',
    'CJ Anderson',
    'CL White',
    'CR Brathwaite',
    'D Wiese',
    'DA Miller',
    'DA Warner',
    'DB Das',
    'DB Ravi Teja',
    'DH Yagnik',
    'DJ Bravo',
    'DJ Hooda',
    'DJ Hussey',
    'DJG Sammy',
    'DL Vettori',
    'DPMD Jayawardene',
    'DR Smith',
    'DS Kulkarni',
    'DT Christian',
    'DW Steyn',
    'EJG Morgan',
    'F du Plessis',
    'G Gambhir',
    'GH Vihari',
    'GJ Bailey',
    'GJ Maxwell',
    'Gurkeerat Singh',
    'HH Pandya',
    'HV Patel',
    'Harbhajan Singh',
    'IK Pathan',
    'Iqbal Abdulla',
    'Ishan Kishan',
    'J Botha',
    'JA Morkel',
    'JC Archer',
    'JC Buttler',
    'JEC Franklin',
    'JH Kallis',
    'JP Duminy',
    'JP Faulkner',
    'K Gowtham',
    'K Rabada',
    'KA Pollard',
    'KC Sangakkara',
    'KD Karthik',
    'KH Pandya',
    'KK Nair',
    'KL Rahul',
    'KM Jadhav',
    'KP Pietersen',
    'KS Williamson',
    'KV Sharma',
    'LR Shukla',
    'LRPL Taylor',
    'M Kartik',
    'M Manhas',
    'M Morkel',
    'MA Agarwal',
    'MA Starc',
    'MC Henriques',
    'MEK Hussey',
    'MF Maharoof',
    'MG Johnson',
    'MK Pandey',
    'MK Tiwary',
    'MM Sharma',
    'MP Stoinis',
    'MR Marsh',
    'MS Dhoni',
    'MS Gony',
    'MV Boucher',
    'Mandeep Singh',
    'Mohammad Nabi',
    'N Pooran',
    'N Rana',
    'NLTC Perera',
    'NV Ojha',
    'OA Shah',
    'P Kumar',
    'P Negi',
    'PJ Cummins',
    'PP Chawla',
    'R Ashwin',
    'R Bhatia',
    'R Dhawan',
    'R Dravid',
    'R Parag',
    'R Sathish',
    'R Sharma',
    'R Tewatia',
    'R Vinay Kumar',
    'RA Jadeja',
    'RA Tripathi',
    'RG Sharma',
    'RJ Harris',
    'RN ten Doeschate',
    'RP Singh',
    'RR Pant',
    'RR Powar',
    'RS Bopara',
    'RV Uthappa',
    'Rashid Khan',
    'S Badrinath',
    'S Dhawan',
    'S Gopal',
    'S Nadeem',
    'S Sreesanth',
    'SA Yadav',
    'SC Ganguly',
    'SE Marsh',
    'SK Raina',
    'SK Trivedi',
    'SK Warne',
    'SL Malinga',
    'SM Curran',
    'SN Khan',
    'SP Narine',
    'SPD Smith',
    'SR Tendulkar',
    'SR Watson',
    'SS Iyer',
    'SS Tiwary',
    'STR Binny',
    'SV Samson',
    'Sachin Baby',
    'Shakib Al Hasan',
    'Shubman Gill',
    'TG Southee',
    'TL Suman',
    'TM Dilshan',
    'TM Head',
    'UT Yadav',
    'V Kohli',
    'V Shankar',
    'VR Aaron',
    'WP Saha',
    'Washington Sundar',
    'Y Nagar',
    'Y Venugopal Rao',
    'YK Pathan',
    'Yuvraj Singh',
    'Z Khan')

    venue=['Barabati Stadium',
    'Brabourne Stadium',
    'Buffalo Park',
    'De Beers Diamond Oval',
    'Dr DY Patil Sports Academy',
    'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
    'Dubai International Cricket Stadium',
    'Eden Gardens',
    'Feroz Shah Kotla',
    'Green Park',
    'Himachal Pradesh Cricket Association Stadium',
    'Holkar Cricket Stadium',
    'JSCA International Stadium Complex',
    'Kingsmead',
    'M Chinnaswamy Stadium',
    'MA Chidambaram Stadium, Chepauk',
    'Maharashtra Cricket Association Stadium',
    'Nehru Stadium',
    'New Wanderers Stadium',
    'Newlands',
    'OUTsurance Oval',
    'Punjab Cricket Association IS Bindra Stadium, Mohali',
    'Punjab Cricket Association Stadium, Mohali',
    'Rajiv Gandhi International Stadium, Uppal',
    'Sardar Patel Stadium, Motera',
    'Saurashtra Cricket Association Stadium',
    'Sawai Mansingh Stadium',
    'Shaheed Veer Narayan Singh International Stadium',
    'Sharjah Cricket Stadium',
    'Sheikh Zayed Stadium',
    "St George's Park",
    'Subrata Roy Sahara Stadium',
    'SuperSport Park',
    'Vidarbha Cricket Association Stadium, Jamtha',
    'Wankhede Stadium']
    def strike_rate(batsman):
        runs=detail[detail['batsman']==batsman]['batsman_runs'].sum()
        balls=detail[detail['batsman']==batsman]['batsman_runs'].count()
        return (runs/balls)*100

    def avg_of_batsman(batsman):
        p=detail['batsman']==batsman
        l=detail['is_wicket']==1
        w=detail[p & l].shape[0]
        r=detail[p]['batsman_runs'].sum()
        return (r/w)

    def death_wickets(bowler):
        a=detail['over']>15
        b=detail['bowler']==bowler
        c=detail['is_wicket']==1
        return detail[a & b & c]['is_wicket'].count()


    def batsman_score(batsman):
        return detail[detail['batsman'] == batsman]['batsman_runs'].sum()

    def death_strike_rate(batsman):
        death = detail[detail['over'] > 15]
        death_batsman = death.groupby('batsman')['batsman_runs'].count()
        x = death.groupby('batsman')['batsman_runs'].count() > 100
        alldeath_batsman = death_batsman[x].index.tolist()
        final = death[death['batsman'].isin(alldeath_batsman)]
        death_runs = final.groupby('batsman')['batsman_runs'].sum()
        death_balls = final.groupby('batsman')['batsman_runs'].count()
        strike= (death_runs / death_balls) * 100
        return strike[batsman]

    def batsman_six(batsman):
        q=detail['batsman']==batsman
        w=detail['batsman_runs']==6
        return detail[q & w].shape[0]

    def batsman_four(batsman):
        q=detail['batsman']==batsman
        w=detail['batsman_runs']==4
        return detail[q & w].shape[0]

    def bowler_wicket(bowler):
        q=detail['bowler']==bowler
        w=detail['is_wicket']==1
        return detail[q & w].shape[0]

    def bowling_avg(bowler):
        a=detail['bowler']==bowler
        b=detail['is_wicket']==1
        runs=detail[a]['batsman_runs'].sum()
        wicket=detail[a & b]['is_wicket'].count()
        return runs/wicket

    def avg_in_venue(batsman,venue):
        a=new['batsman']==batsman
        b=new['venue']==venue
        c=new['is_wicket']==1
        runs=new[a & b]['batsman_runs'].sum()
        wicket=new[a & b & c]['is_wicket'].count()
        return runs/wicket

    def batsman_against_bowler(batsman,bowler):
        a=detail['batsman']==batsman
        b=detail['bowler']==bowler
        c=detail['is_wicket']==1
        runs=detail[a & b]['batsman_runs'].sum()
        wicket=detail[a & b & c]['is_wicket'].count()
        st.write ('The',batsman,'score runs in front of',bowler,'are',runs,'and The',bowler,'dismissed him',wicket)


    st.title('IPl Data Analysis 2008-2020')
    st.sidebar.header('Select the player performance attribute to measure')
    select=st.sidebar.radio('',('Batsman Strike Rate','Batsman average','Death Over Strike Rate','Batsman Average in Venue','Total Runs','Batsman vs Bowler','Number of sixes','Number of fours','Total Wickets','Death Over Wickets','Bowling Average'))

    if select=='Batsman Strike Rate':
        a=st.selectbox('Select Player Name',batsman)
        if st.button('Calculate'):
            st.write('Strike rate of',a,'is',strike_rate(a))
    elif select=='Batsman average':
        a = st.selectbox('Select Player Name',batsman)
        if st.button('Calculate'):
            st.write('Average of', a, 'is', avg_of_batsman(a))

    elif select=='Death Over Wickets':
        a = st.selectbox('Select Player Name',bowler)
        if st.button('Calculate'):
            st.write('Total Death Over Wickets of', a, 'is', death_wickets(a))

    elif select=='Total Runs':
        a = st.selectbox('Select Player Name',batsman)
        if st.button('Calculate'):
            st.write('Total Runs Score by', a, 'is', batsman_score(a))


    elif select=='Death Over Strike Rate':
        st.write('You Can Choose Only Those Players Who Played at Least 50 balls in Death Over')
        a = st.selectbox('Select Player Name',death_players)
        if st.button('Calculate'):
                st.write('Strike Rate in Death Overs of', a, 'is', death_strike_rate(a))

    elif select=='Number of sixes':
        a = st.selectbox('Select Player Name',batsman)
        if st.button('Calculate'):
            st.write('Total Sixes by', a, 'is', batsman_six(a))


    elif select=='Number of fours':
        a = st.selectbox('Select Player Name',batsman)
        if st.button('Calculate'):
            st.write('Total Fours by', a, 'is', batsman_four(a))

    elif select=='Total Wickets':
        a = st.selectbox('Select Player Name',bowler)
        if st.button('Calculate'):
            st.write('Total Wickets by', a, 'is', bowler_wicket(a))

    elif select=='Bowling Average':
        a = st.selectbox('Select Player Name',bowler)
        if st.button('Calculate'):
            st.write('Bowling Average of ', a, 'is', bowling_avg(a))

    elif select=='Batsman Average in Venue':
        a = st.selectbox('Select Player Name',batsman)
        b = st.selectbox('Select Venue',venue)
        if st.button('Calculate'):
            st.write(a,'Average in',b,'is',avg_in_venue(a,b))

    elif select=='Batsman vs Bowler':
        a = st.selectbox('Select Batsman',batsman)
        b = st.selectbox('Select Bowler',bowler)
        if st.button('Calculate'):
            batsman_against_bowler(a,b)




with prediction_tab:
    
    st.title("IPL Prediction Engine ")
    data=pd.read_csv('https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/final_output.csv')
    gw_list=data['date'].unique().tolist()
    menu=gw_list

    st.sidebar.header('Select the Game Date for match prediction(2021)')
    choice = st.sidebar.selectbox('Date:',menu)  
    data_fil=data.loc[data.date==choice]
    data_fil=data_fil.reset_index(drop=True)
#  st.write(data_fil)

    for i in range(data_fil.shape[0]):
        st.subheader('MATCH OF THE DAY')

        team_x=data_fil.loc[i,'team_name_x']
        team_y=data_fil.loc[i,'team_name_y']

        team_x=team_x.replace(" ",'%20')
        team_y=team_y.replace(" ",'%20')
    
        pred_x=data_fil.loc[i,'pred_team_x']
        pred_y=data_fil.loc[i,'pred_team_y']
    
        col= st.columns(2)

        img_path='https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/Logos/'+team_x+'.jpg'
        img=io.imread(img_path)  
        img_path='https://raw.githubusercontent.com/arpitsolanki/IPL-Prediction-Engine/main/Logos/'+team_y+'.jpg'
        img1=io.imread(img_path)  
    
        col[0].image(img,width=250)
        col[1].image(img1,width=250)
    
        var="VENUE"
        st.markdown("<h3 style='text-align: center; color: white;'>"+var+"</h3>", unsafe_allow_html=True)

        var=data_fil.loc[i,'venue']
        st.markdown("<h4 style='text-align: center; color: white;'>"+var+"</h4>", unsafe_allow_html=True)

        var="WIN PROBABILITY"
        st.markdown("<h3 style='text-align: center; color: white;'>"+var+"</h3>", unsafe_allow_html=True)

  #  st.subheader('WIN PROBABILITY')
        col= st.columns((1,2,1,2))
        col[1].header("{0:.0%}".format(pred_x))
        col[3].header("{0:.0%}".format(pred_y))

        var="WINNING TEAM"
        st.markdown("<h3 style='text-align: center; color: white;'>"+var+"</h3>", unsafe_allow_html=True)

        var=data_fil.loc[i,'winning_team']
        st.markdown("<h4 style='text-align: center; color: white;'>"+var+"</h4>", unsafe_allow_html=True)


