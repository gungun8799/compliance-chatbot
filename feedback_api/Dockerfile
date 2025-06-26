# feedback_api/Dockerfile

FROM node:18-alpine

WORKDIR /usr/src/app

# copy package.json + lockfile, install deps
COPY package*.json ./
RUN npm install --production

# copy your Express code
COPY server.js . 

EXPOSE 4001

CMD ["node", "server.js"]