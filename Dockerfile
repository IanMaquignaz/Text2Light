# StyleGAN3
FROM nvcr.io/nvidia/pytorch:21.08-py3
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
RUN pip install imageio imageio-ffmpeg==0.4.4 pyspng==0.1.0 pandas
WORKDIR /workspace

# Update the box (refresh apt-get)
# Packages can be messed up, but using '--fix-missing' results in a full docker rebuild
# This uses '--fix-missing' only if something fails.
RUN apt-get update -y --fix-missing
RUN apt-get update -y && if [ $? -ne 0 ] ; then apt-get update -y --fix-missing ; fi ;

# Dependencies
RUN apt-get install libxrender-dev libxext6 libsm6 libglib2.0-0 -y

# Creates a non-root user with an explicit UID
ARG USER_NAME="toor"
ARG USER_ID=5678
ARG GROUP_ID=8765
RUN groupadd -g ${GROUP_ID} docker 
RUN useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME}
RUN echo "${USER_NAME}:toor" |  chpasswd 
USER $USER_ID:${GROUP_ID}