import pulp
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import base64

import fastprogress
import sys
sys.path.append("..")

col1, col2, col3 = st.beta_columns([1,6,2])

with col1:
	st.write("")
	

with col2:
	st.title('Final Year Project')
	st.markdown("""---""")
	st.markdown('''
	    <a href="https://share.streamlit.io/robinraj1999/fpl-optimal-team-optimizer/main/mywork1.py">
	        <img src="https://www.chapsandco.ae/wp-content/uploads/FantasyLaunch-TakeoverEditorial.jpg" width="400" />
	    </a>''',
	    unsafe_allow_html=True
	)
	
	st.markdown("""---""")
   
	st.markdown('''
	    <a href="https://www.google.com">
	        <img src="https://resources.premierleague.com/premierleague/photo/2020/06/08/e622b94f-6ad4-43d4-a1c2-226eb8b2f163/FantasyLaunch-EditorialLead-Graphic1-002-.png" width="400" />
	    </a>''',
	    unsafe_allow_html=True
	)
	
with col3:
	st.write("")
	st.markdown('#')	
	st.markdown('#')
	st.markdown('#')
	st.title('Generate Optimal Team')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	st.markdown('#')
	
	st.title('Choose Team Manually')

   