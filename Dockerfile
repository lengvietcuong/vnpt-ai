# Dump build stage - preload the database
FROM pgvector/pgvector:pg17-bookworm AS dumper

# Copy the database dump file
COPY data/data.dump /tmp/data.dump

# Modify the entrypoint to skip starting the PostgreSQL daemon
# This allows us to initialize the database without running the server
RUN sed -i "s/exec \"\$@\"/echo \"skipping...\"/g" /usr/local/bin/docker-entrypoint.sh

# Set environment variables for database initialization
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=vn_knowledge
ENV PGDATA=/data

# Run the entrypoint to initialize the database structure
# This creates the base cluster and the vn_knowledge database
RUN /usr/local/bin/docker-entrypoint.sh postgres

# Start PostgreSQL temporarily to restore the dump, then stop it
# The dump file is in custom format, so we use pg_restore
# Must run as postgres user, not root
# Use --no-owner --no-privileges to avoid role ownership issues
RUN chown -R postgres:postgres /data /tmp/data.dump && \
   su postgres -c "pg_ctl -D /data -o \"-c listen_addresses=''\" -w start" && \
   su postgres -c "pg_restore -U postgres -d vn_knowledge --no-owner --no-privileges -v /tmp/data.dump || echo 'Restore completed with warnings'" && \
   su postgres -c "pg_ctl -D /data -m fast -w stop"

# Final build stage
FROM pgvector/pgvector:pg17-bookworm

# Set PGDATA to match the dumper stage
ENV PGDATA=/var/lib/postgresql/data/pgdata

# Copy the preloaded database from the dumper stage
COPY --from=dumper /data $PGDATA

# Ensure proper ownership of the database directory
RUN chown -R postgres:postgres /var/lib/postgresql/data

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
   python3 \
   python3-pip \
   postgresql-client \
   supervisor \
   && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy application code and install Python dependencies
COPY requirements.txt /code/
RUN pip3 install --no-cache-dir --upgrade pip --break-system-packages && \
   pip3 install --no-cache-dir -r requirements.txt --break-system-packages

COPY . /code/

# Create supervisor configuration directory
RUN mkdir -p /var/log/supervisor /etc/supervisor/conf.d

# Create supervisor configuration file to run PostgreSQL and the application
RUN echo '[supervisord]\n\
   nodaemon=true\n\
   user=root\n\
   \n\
   [supervisorctl]\n\
   serverurl=unix:///var/run/supervisor.sock\n\
   \n\
   [unix_http_server]\n\
   file=/var/run/supervisor.sock\n\
   \n\
   [rpcinterface:supervisor]\n\
   supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface\n\
   \n\
   [program:postgresql]\n\
   command=/usr/local/bin/docker-entrypoint.sh postgres\n\
   autostart=true\n\
   autorestart=true\n\
   stdout_logfile=/dev/stdout\n\
   stdout_logfile_maxbytes=0\n\
   stderr_logfile=/dev/stderr\n\
   stderr_logfile_maxbytes=0\n\
   environment=POSTGRES_USER=postgres,POSTGRES_PASSWORD=postgres,POSTGRES_DB=vn_knowledge,PGDATA=/var/lib/postgresql/data/pgdata\n\
   \n\
   [program:app]\n\
   command=python3 /code/predict.py\n\
   autostart=false\n\
   autorestart=false\n\
   stdout_logfile=/dev/stdout\n\
   stdout_logfile_maxbytes=0\n\
   stderr_logfile=/dev/stderr\n\
   stderr_logfile_maxbytes=0\n\
   startsecs=0\n\
   priority=999' > /etc/supervisor/conf.d/supervisord.conf

# Create startup script that waits for PostgreSQL before starting the app
RUN echo '#!/bin/bash\n\
   set -e\n\
   \n\
   # Start supervisor in the background\n\
   /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf &\n\
   \n\
   # Wait for PostgreSQL to be ready\n\
   echo "Waiting for PostgreSQL to be ready..."\n\
   until pg_isready -h localhost -U postgres; do\n\
   sleep 1\n\
   done\n\
   echo "PostgreSQL is ready!"\n\
   \n\
   # Start the application\n\
   supervisorctl start app\n\
   \n\
   # Wait for the app to finish\n\
   while supervisorctl status app | grep -q RUNNING; do\n\
   sleep 1\n\
   done\n\
   \n\
   # Get the exit code of the app\n\
   APP_EXIT_CODE=$(supervisorctl status app | grep -oP "exit status \\K[0-9]+" || echo "0")\n\
   \n\
   # Stop everything\n\
   supervisorctl stop all\n\
   \n\
   # Exit with the app exit code\n\
   exit $APP_EXIT_CODE' > /code/start.sh && \
   chmod +x /code/start.sh

CMD ["/code/start.sh"]