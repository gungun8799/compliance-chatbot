select distinct (u."identifier") as employee_id
from "Thread" t

inner join "Step" s  
on (t.id = s."threadId")

inner join "User" u
on (t."userId" = u.id)

where u."identifier" is not null
      and s."type" in ('assistant_message', 'user_message')
      and s."createdAt" between %(start_date)s and %(end_date)s
