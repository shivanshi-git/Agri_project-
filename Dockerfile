# Step 1: Use official Python image
FROM python:3.10-slim

# Step 2: Set working directory
WORKDIR /app

# Step 3: Copy requirements file into container
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy everything (including templates) into container
COPY . /app

# Step 6: Verify templates directory exists (debug step)
RUN ls -R /app

# Step 7: Expose port
EXPOSE 5000

# Step 8: Run app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
