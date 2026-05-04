import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load models
classifier       = joblib.load("notebooks\models/classifier.pkl")
anomaly_detector = joblib.load("notebooks\models/anomaly_detector.pkl")
scaler           = joblib.load("notebooks\models/scaler.pkl")
feature_names    = joblib.load("notebooks\models/feature_names.pkl")

# App title
st.title("🛡️ Smart API Protection")
st.write("Enter network flow details to predict whether the traffic is **Normal** or an **Attack**.")

st.markdown("---")

# ── Input fields ──────────────────────────────────────────────────────────────
st.subheader("Network Flow Features")

col1, col2, col3 = st.columns(3)

with col1:
    dur               = st.number_input("Duration (dur)", min_value=0.0, value=0.0, format="%.6f", help="Record total duration")
    sbytes            = st.number_input("Source Bytes (sbytes)", min_value=0, value=0, help="Source to destination bytes")
    dbytes            = st.number_input("Destination Bytes (dbytes)", min_value=0, value=0, help="Destination to source bytes")
    sttl              = st.number_input("Source TTL (sttl)", min_value=0, max_value=255, value=64, help="Source to destination time to live")
    dttl              = st.number_input("Destination TTL (dttl)", min_value=0, max_value=255, value=64, help="Destination to source time to live")
    sloss             = st.number_input("Source Loss (sloss)", min_value=0, value=0, help="Source packets retransmitted or dropped")
    dloss             = st.number_input("Destination Loss (dloss)", min_value=0, value=0, help="Destination packets retransmitted or dropped")
    sload             = st.number_input("Source Load (sload)", min_value=0.0, value=0.0, format="%.4f", help="Source bits per second")
    dload             = st.number_input("Destination Load (dload)", min_value=0.0, value=0.0, format="%.4f", help="Destination bits per second")
    spkts             = st.number_input("Source Packets (spkts)", min_value=0, value=1, help="Source to destination packet count")
    dpkts             = st.number_input("Destination Packets (dpkts)", min_value=0, value=1, help="Destination to source packet count")
    swin              = st.number_input("Source Window (swin)", min_value=0, max_value=65535, value=255, help="Source TCP window advertisement")
    dwin              = st.number_input("Destination Window (dwin)", min_value=0, max_value=65535, value=255, help="Destination TCP window advertisement")

with col2:
    smeansz           = st.number_input("Src Mean Pkt Size (smeansz)", min_value=0, value=0, help="Mean of flow packet sizes transmitted by source")
    dmeansz           = st.number_input("Dst Mean Pkt Size (dmeansz)", min_value=0, value=0, help="Mean of flow packet sizes transmitted by destination")
    trans_depth       = st.number_input("Trans Depth (trans_depth)", min_value=0, value=0, help="Pipelined depth into the connection")
    res_bdy_len       = st.number_input("Response Body Length (res_bdy_len)", min_value=0, value=0, help="Actual uncompressed content size of HTTP response body")
    sjit              = st.number_input("Source Jitter (sjit)", min_value=0.0, value=0.0, format="%.4f", help="Source jitter in milliseconds")
    djit              = st.number_input("Destination Jitter (djit)", min_value=0.0, value=0.0, format="%.4f", help="Destination jitter in milliseconds")
    sintpkt           = st.number_input("Src Inter-Pkt Time (sintpkt)", min_value=0.0, value=0.0, format="%.4f", help="Source interpacket arrival time in milliseconds")
    dintpkt           = st.number_input("Dst Inter-Pkt Time (dintpkt)", min_value=0.0, value=0.0, format="%.4f", help="Destination interpacket arrival time in milliseconds")
    tcprtt            = st.number_input("TCP RTT (tcprtt)", min_value=0.0, value=0.0, format="%.6f", help="TCP connection setup round-trip time")
    synack            = st.number_input("SYN-ACK Time (synack)", min_value=0.0, value=0.0, format="%.6f", help="Time between SYN and SYN-ACK")
    ackdat            = st.number_input("ACK-Data Time (ackdat)", min_value=0.0, value=0.0, format="%.6f", help="Time between SYN-ACK and ACK")
    is_sm_ips_ports   = st.selectbox("Same IPs & Ports (is_sm_ips_ports)", [0, 1], help="1 if source and destination IPs and ports are equal")

with col3:
    ct_state_ttl      = st.number_input("CT State TTL (ct_state_ttl)", min_value=0, value=0, help="No. connections with same state and TTL")
    ct_flw_http_mthd  = st.number_input("CT HTTP Method (ct_flw_http_mthd)", min_value=0, value=0, help="No. flows using the same HTTP method")
    is_ftp_login      = st.selectbox("FTP Login (is_ftp_login)", [0, 1], help="1 if the FTP session is accessed by user and password")
    ct_ftp_cmd        = st.number_input("CT FTP Commands (ct_ftp_cmd)", min_value=0, value=0, help="No. of FTP commands in the session")
    ct_srv_src        = st.number_input("CT Srv Src (ct_srv_src)", min_value=0, value=0, help="No. connections same service and src address in last 100")
    ct_srv_dst        = st.number_input("CT Srv Dst (ct_srv_dst)", min_value=0, value=0, help="No. connections same service and dst address in last 100")
    ct_dst_ltm        = st.number_input("CT Dst LTM (ct_dst_ltm)", min_value=0, value=0, help="No. connections same dst address in last 100")
    ct_src_ltm        = st.number_input("CT Src LTM (ct_src_ltm)", min_value=0, value=0, help="No. connections same src address in last 100")
    ct_src_dport_ltm  = st.number_input("CT Src Dst Port LTM (ct_src_dport_ltm)", min_value=0, value=0, help="No. connections same src address and dst port in last 100")
    ct_dst_sport_ltm  = st.number_input("CT Dst Src Port LTM (ct_dst_sport_ltm)", min_value=0, value=0, help="No. connections same dst address and src port in last 100")
    ct_dst_src_ltm    = st.number_input("CT Dst Src LTM (ct_dst_src_ltm)", min_value=0, value=0, help="No. connections same src and dst address in last 100")

st.markdown("---")

# ── Build input dataframe ─────────────────────────────────────────────────────
input_data = pd.DataFrame({
    "dur":              [dur],
    "sbytes":           [sbytes],
    "dbytes":           [dbytes],
    "sttl":             [sttl],
    "dttl":             [dttl],
    "sloss":            [sloss],
    "dloss":            [dloss],
    "sload":            [sload],
    "dload":            [dload],
    "spkts":            [spkts],
    "dpkts":            [dpkts],
    "swin":             [swin],
    "dwin":             [dwin],
    "smeansz":          [smeansz],
    "dmeansz":          [dmeansz],
    "trans_depth":      [trans_depth],
    "res_bdy_len":      [res_bdy_len],
    "sjit":             [sjit],
    "djit":             [djit],
    "sintpkt":          [sintpkt],
    "dintpkt":          [dintpkt],
    "tcprtt":           [tcprtt],
    "synack":           [synack],
    "ackdat":           [ackdat],
    "is_sm_ips_ports":  [is_sm_ips_ports],
    "ct_state_ttl":     [ct_state_ttl],
    "ct_flw_http_mthd": [ct_flw_http_mthd],
    "is_ftp_login":     [is_ftp_login],
    "ct_ftp_cmd":       [ct_ftp_cmd],
    "ct_srv_src":       [ct_srv_src],
    "ct_srv_dst":       [ct_srv_dst],
    "ct_dst_ltm":       [ct_dst_ltm],
    "ct_src_ltm":       [ct_src_ltm],
    "ct_src_dport_ltm": [ct_src_dport_ltm],
    "ct_dst_sport_ltm": [ct_dst_sport_ltm],
    "ct_dst_src_ltm":   [ct_dst_src_ltm],
})

# Keep only features the model was trained on
input_data = input_data[[f for f in feature_names if f in input_data.columns]]

# ── Data preview ──────────────────────────────────────────────────────────────
st.subheader("Input Data Preview")
st.dataframe(input_data)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)

    threat_score = classifier.predict_proba(input_scaled)[0][1]
    anomaly_flag = anomaly_detector.predict(input_scaled)[0] == -1

    if threat_score > 0.85 or anomaly_flag:
        decision = "BLOCK"
    elif threat_score > 0.5:
        decision = "FLAG"
    else:
        decision = "ALLOW"

    st.success(f"Predicted Decision: **{decision}**  |  Threat Score: `{threat_score:.4f}`  |  Anomaly: {'Yes 🚨' if anomaly_flag else 'No ✅'}")

    st.subheader("Insights & Suggestions")

    if decision == "BLOCK":
        st.error("🚫 High threat detected. Immediately block the source IP and investigate the flow.")
    elif decision == "FLAG":
        st.warning("⚠️ Suspicious traffic. Monitor this connection closely and consider rate limiting.")
    else:
        st.success("✅ Traffic appears normal. No significant threat indicators found.")
