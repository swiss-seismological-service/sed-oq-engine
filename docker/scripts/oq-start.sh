#!/bin/bash
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2019-2025 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

export PYTHONPATH=$HOME

echo "Starting OpenQuake Engine DbServer..."
oq dbserver start 2>&1 &
DBSERVER_PID=$!

echo "Waiting for DbServer to become ready (PID: $DBSERVER_PID)..."
MAX_WAIT=900
WAIT_COUNT=0
while :
do
    if ! kill -0 $DBSERVER_PID 2>/dev/null; then
        echo "ERROR: DbServer process died unexpectedly!"
        wait $DBSERVER_PID
        exit 1
    fi

    (echo > /dev/tcp/localhost/1908) >/dev/null 2>&1
    result=$?
    if [[ $result -eq 0 ]]; then
        echo "DbServer is ready on port 1908"
        break
    fi

    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -gt $MAX_WAIT ]; then
        echo "ERROR: DbServer failed to start within $MAX_WAIT seconds"
        exit 1
    fi
    sleep 1
done

if [ "$LOCKDOWN" = "True" ]; then
    echo "LOCKDOWN mode enabled - setting up authentication..."
    echo "LOCKDOWN = True" > $HOME/local_settings.py

    export DJANGO_SETTINGS_MODULE=openquake.server.settings

    echo "Running database migrations..."
    python3 -m openquake.server.manage migrate 2>&1

    if [ -n "$OQ_ADMIN_LOGIN" ]; then
        echo "Creating superuser: ${OQ_ADMIN_LOGIN}..."
        python3 -m openquake.server.manage shell -c "from django.contrib.auth.models import User; User.objects.filter(username='${OQ_ADMIN_LOGIN}').exists() or User.objects.create_superuser('${OQ_ADMIN_LOGIN}', '${OQ_ADMIN_EMAIL}', '${OQ_ADMIN_PASSWORD}')" 2>&1 || echo "User may already exist"
    else
        echo "Creating default superuser: admin..."
        python3 -m openquake.server.manage shell -c "from django.contrib.auth.models import User; User.objects.filter(username='admin').exists() or User.objects.create_superuser('admin', 'admin@example.com', 'admin')" 2>&1 || echo "User may already exist"
    fi
fi

if [ -t 1 ]; then
    echo "Starting WebUI in TTY mode..."
    exec oq webui start 0.0.0.0:8800 -s &>> $HOME/oqdata/webui.log &
    /bin/bash
else
    echo "Starting WebUI in headless mode..."
    exec oq webui start 0.0.0.0:8800 -s
fi
