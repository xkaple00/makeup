FROM python:3.8

# RUN apt update
# RUN apt install -y python3-dev gcc

# ADD requirements.txt requirements.txt
# ADD demo_st.py demo_st.py

# # Install required libraries
# RUN pip install -r requirements.txt
# COPY . /app

# EXPOSE 8008

# CMD streamlit run app.py

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# RUN apt install -y libprotobuf-dev protobuf-compiler 
RUN apt-get update && apt-get -y install cmake protobuf-compiler &&\
    apt install -y gcc &&\
    apt-get install -y libgl1-mesa-dev &&\
    apt install libgl1-mesa-glx  &&\
    pip install -r requirements.txt

# Install dependencies
# RUN pip install -r requirements.txt

# copying all files over
COPY . /app

# Expose port 
ENV PORT 8501

# cmd to launch app when container is run
CMD streamlit run demo_st.py

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit

RUN apt-get install 'libsm6'\ 
    'libxext6'  -y

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'