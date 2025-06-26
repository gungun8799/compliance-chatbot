#!/bin/sh
npx prisma migrate deploy
npx prisma studio &
wait