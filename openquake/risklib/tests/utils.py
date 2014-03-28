# coding=utf-8
# Copyright (c) 2010-2014, GEM Foundation.
#
# OpenQuake Risklib is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# OpenQuake Risklib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with OpenQuake Risklib. If not, see
# <http://www.gnu.org/licenses/>.

import os
import csv
import collections


def new(tuple_type, **kwargs):
    '''Instantiate a namedtuple class with missing fields defaulting to None
    >>> Point = namedtuple('Point', 'x y z')
    >>> new(Point, x=1, y=2)
    Point(x=1, y=2, z=None)
    '''
    params = dict.fromkeys(tuple_type._fields)
    params.update(kwargs)
    return tuple_type(**params)


def vectors_from_csv(name, dirname):
    "Read columns of floats as an array for a .csv file with an header"
    fullname = os.path.join(dirname, name + '.csv')
    with open(fullname) as f:
        reader = csv.reader(f)
        fields = reader.next()
        ntclass = collections.namedtuple(name, fields)
        columns = zip(*[map(float, row) for row in reader])
    return ntclass(*columns)
