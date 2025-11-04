# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# from .DSRV import *
# from .frigate import *
# from .otter import *
# from .ROVzefakkel import *
# from .semisub import *
# from .shipClarke83 import *
# from .supply import *
# from .tanker import *
# from .remus100 import *
# from .torpedo import *


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vehicles package
"""

# ✅ 导入基础车辆
from .otter import otter

# ✅ 导入扩展车辆
from .otter_station_keeping import OtterStationKeeping

__all__ = [
    'otter',
    'OtterStationKeeping',
]