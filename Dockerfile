# Python support can be specified down to the minor or micro version
# (e.g. 3.6 or 3.6.3).
# OS Support also exists for jessie & stretch (slim and full).
# See https://hub.docker.com/r/library/python/ for all supported Python
# tags from Docker Hub.
FROM hdgigante/opencv:fedora-opencv4

# If you prefer miniconda:
#FROM continuumio/miniconda3

LABEL Name=plate_rec Version=0.0.1
EXPOSE 5000

WORKDIR /app
ADD . /app
COPY ./pip.conf /etc/pip.conf


# Using pip:
RUN python3 -m pip install -r requirements.txt
CMD ["python3", "image_tfserving.py"]

# Using pipenv:
#RUN python3 -m pip install pipenv
#RUN pipenv install --ignore-pipfile
#CMD ["pipenv", "run", "python3", "-m", "plate_rec"]

# Using miniconda (make sure to replace 'myenv' w/ your environment name):
#RUN conda env create -f environment.yml
#CMD /bin/bash -c "source activate myenv && python3 -m plate_rec"
