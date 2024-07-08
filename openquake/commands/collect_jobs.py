# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2024 GEM Foundation
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

import sys
import time
import numpy
from openquake.hazardlib.site import SiteCollection
from openquake.commonlib import logs, datastore


def collect(job_ids, dstore):
    """
    Collect from job_ids and save the concatenated arrays in dstore
    """
    sitecols = []
    dic = {'performance_data': [], 'hcurves-stats': [], 'hmaps-stats': []}
    for job_id in job_ids:
        with datastore.read(job_id) as ds:
            arr = ds['sitecol'].array
            fields = sorted(arr.dtype.names)
            sitecols.append(arr[fields])
            for name in dic:
                if name in ds:
                    dic[name].append(ds[name][:])

    last = datastore.read(job_ids[-1])
    sitecol = object.__new__(SiteCollection)
    sitecol.array = numpy.concatenate(sitecols)
    dstore['sitecol'] = sitecol
    dstore['oqparam'] = last['oqparam']
    if 'source_info' in last:
        dstore['source_info'] = last['source_info'][:]
    if 'weights' in last:
        dstore['weights'] = last['weights'][:]
    if 'full_lt' in last:
        dstore['full_lt'] = last['full_lt']
    for key in ('engine_version', 'date', 'checksum32'):
        dstore['/'].attrs[key] = last['/'].attrs[key]
    for name, arrays in dic.items():
        if arrays:
            dstore[name] = numpy.concatenate(arrays)
            js = last[name].attrs.get('json')
            if js:
                dstore[name].attrs['json'] = js


def main(job_ids: int):
    """
    Wait for the given jobs to finish and then collect the results
    """
    while True:
        rows = logs.dbcmd('SELECT id, status FROM job WHERE id in (?X)', job_ids)
        # print(rows)
        failed = [row for row in rows if row.status == 'failed']
        if failed:
            sys.exit('Job %d failed' % failed[0].id)
        complete = [row for row in rows if row.status == 'complete']
        if len(complete) == len(rows):
            dstore, log = datastore.build_dstore_log('-'.join(map(str, job_ids)))
            with dstore, log:
                collect(job_ids, dstore)
            print('All jobs completed correctly, saved result in', dstore.filename)
            break
        time.sleep(30.)

main.job_ids = dict(help='number of jobs to create in the database', nargs='+')
