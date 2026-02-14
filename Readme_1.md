\# ML-Assisted Sheet Metal Bending (FEM-Calibrated Digital Twin)



This project presents a data-driven digital twin for V-bending of sheet metal,

trained on FEM (LS-DYNA) simulation data and deployed as an interactive web application.



The system replaces time-intensive finite element simulations with

instant, FEM-calibrated surrogate model calculations.



\## What This Tool Does



\- Computes stable bend geometry using FEM-trained ML models

\- Solves the inverse problem to determine required control parameters

\- Visualizes bending profiles and geometric deviations

\- Generates robot-relevant toolpath parameters



No physical time-sequence or springback narration is used in the UI.

All results are presented as stable geometric outcomes.



\## How to Run 



This application is deployed using Streamlit Cloud.



👉 \*\*Launch link:\*\*  

https://ml-sheet-metal-bending.streamlit.app/



(No local installation required.)



\## Repository Structure



\- `app/` – Streamlit UI and assets

\- `data/` – FEM-derived dataset as csv and all the files used to  extract the required data

\- `models/` – Trained ML surrogate models

\- `toolpath/` – Toolpath computation logic

\- `tools/` – Dataset generation utilities

\- `utils/` – Plotting and visualization helpers




All models are trained on FEM data and the full pipeline is transparent and reproducible.



