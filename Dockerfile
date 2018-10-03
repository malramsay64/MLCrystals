FROM continuumio/miniconda3:4.5.4

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ENV NB_UID 1000
ENV NB_USER jovyan
ENV HOME /home/$NB_USER
ENV SHELL /bin/bash

# Create jovyan user with UID=1000 and in the 'users' group
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

# Allow all users nopasswd sudo access
RUN echo "ALL ALL = (ALL) NOPASSWD: ALL" >> /etc/sudoers

USER $NB_UID

WORKDIR /home/$NB_USER

COPY environment-lock.txt environment-lock.txt
RUN conda create --name MLCrystals --file environment-lock.txt

RUN mkdir /home/$NB_USER/work
WORKDIR /home/$NB_USER/work

RUN echo "source activate MLCrystals" >> ~/.bashrc

ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888/tcp

CMD ["/home/jovyan/.conda/envs/MLCrystals/bin/jupyter", "notebook", "--ip=*", "--port=8888", "--no-browser"]
