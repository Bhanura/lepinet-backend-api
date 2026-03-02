# Python 3.10 භාවිතය
FROM python:3.10-slim

# Working directory එක සකස් කිරීම
WORKDIR /code

# Requirements file එක copy කර libraries install කිරීම
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Hugging Face Spaces සඳහා root නොවන user කෙනෙක් සෑදීම (ආරක්ෂාවට)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# කේතය copy කිරීම
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# FastAPI server එක port 7860 හරහා run කිරීම (HF spaces වල default port එක)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]