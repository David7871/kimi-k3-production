# Usage: grid.py SQLITE PATTERN... -> most common grid/block dims of kernels whose short name matches each LIKE pattern
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
for pat in sys.argv[2:]:
    for row in c.execute("select s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
                         "join StringIds s on s.id = k.shortName where s.value like ? group by 1,2,3,4,5,6 order by 7 desc limit 3", (pat,)):
        print(sys.argv[1].split("/")[-3], row[0][:45], "grid", row[1:4], "block", row[4:6], "n", row[6])
