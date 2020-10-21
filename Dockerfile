FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

LABEL maintainer="Angel Sevilla Camins <a.sevillag@anchormen.nl>"

# From https://hub.docker.com/r/rocker/r-ubuntu/dockerfile

ENV R_VERSION=4.0.3
ENV S6_VERSION=v1.21.7.0
ENV RSTUDIO_VERSION=latest
ENV PATH=/usr/lib/rstudio-server/bin:$PATH

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		software-properties-common \
        dirmngr \
        ed \
		less \
		locales \
		vim-tiny \
		wget \
		ca-certificates \
        && add-apt-repository --enable-source --yes "ppa:marutter/rrutter4.0" \
        && add-apt-repository --enable-source --yes "ppa:c2d4u.team/c2d4u4.0+"

## Configure default locale, see https://github.com/rocker-org/rocker/issues/19
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

## This was not needed before but we need it now
ENV DEBIAN_FRONTEND noninteractive

## Otherwise timedatectl will get called which leads to 'no systemd' inside Docker
ENV TZ UTC

# Now install R and keras. Configure R to use default python installation 
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base=${R_VERSION}* \
        r-base-dev=${R_VERSION}* \
        r-recommended=${R_VERSION}* \
  	&& echo 'Sys.setenv(RETICULATE_PYTHON="/usr/local/bin/python")' >> /etc/R/Rprofile.site \
    && apt -y install r-cran-devtools r-cran-keras r-cran-ggplot2 \
 	&& rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
 	&& rm -rf /var/lib/apt/lists/*

# Use R within python
RUN pip install rpy2

# Install R-studio
COPY scripts /scripts
RUN /scripts/install_rstudio.sh
RUN /scripts/install_pandoc.sh
EXPOSE 8787

## Set up Jupyter init script to use S6 
RUN mkdir -p /etc/services.d/jupyter
RUN echo $'#!/usr/bin/with-contenv bash \n\
source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root' > /etc/services.d/jupyter/run

CMD ["/init"]

