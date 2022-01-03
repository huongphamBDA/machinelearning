#FROM ubuntu
FROM python:3.8

ENV APP_HOME /app
WORKDIR $APP_HOME

#RUN apt-get update \
#    && apt-get install --no-install-recommends --yes build-essential libmariadb3 libmariadb-dev mysql-client python3 python3-pip python3-dev \
#    && apt-get autoremove -y

RUN apt-get update \
    && apt-get install --no-install-recommends --yes mariadb-client libmariadb3 libmariadb-dev build-essential \
    && apt-get autoremove -y

COPY final_project.sh ./final_project.sh
COPY mariadb_data/baseball.sql ./baseball.sql
COPY utils_pham.py ./utils_pham.py
COPY process_baseball_pham.py ./process_baseball_pham.py
COPY final_project_pham.py ./final_project_pham.py
COPY final_project.sql ./final_project.sql
COPY requirements.txt ./requirements.txt

RUN chmod +x ./final_project.sh
RUN mkdir ./output
RUN pip3 install -r requirements.txt

CMD ./final_project.sh