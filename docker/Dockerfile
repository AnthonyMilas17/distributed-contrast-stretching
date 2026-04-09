FROM ubuntu:latest
#
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install gcc -y
RUN apt-get install g++ -y
RUN apt-get install build-essential -y
RUN apt-get install valgrind -y
RUN apt-get install vim -y
RUN apt-get install cmake -y
RUN apt install openmpi-bin openmpi-common libopenmpi-dev -y
RUN apt install python3 -y
#
# turn off history file creation:
#
RUN echo "export HISTFILE=/dev/null" >> /etc/profile
#
# add a user (with no pwd) so we don't run as root:
#
RUN useradd -m -g users user
