# 베이스 이미지로 Python 3.8을 사용합니다.
FROM python:3.8-slim

# 작업 디렉터리를 설정합니다.
WORKDIR /app

# 필요한 패키지 목록을 복사합니다.
COPY requirements.txt .

# 필요한 패키지를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉터리의 모든 파일을 컨테이너의 작업 디렉터리로 복사합니다.
COPY . .

# Flask 애플리케이션을 실행합니다.
CMD ["python", "app.py"]
