FROM debian

# To make logging/io work w/Docker
ENV PYTHONUNBUFFERED=1

# Get basic python dependencies
RUN apt update && apt upgrade -y && \
    apt install python3-pip curl nano -y && \
    pip3 install pandas numpy scikit-learn scipy && \
    pip3 install -i https://test.pypi.org/simple/ lambdata-worldwidekatie==1.3 && \
    python3 -c "import lambdata-worldwidekatie==1.3"