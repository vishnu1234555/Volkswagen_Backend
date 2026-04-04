import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 1000,
  duration: '30s',
};

export default function () {
  const url = 'http://localhost:8000/retrieve';
  const payload = JSON.stringify({
    question: 'What are the specs of the Taigun GT?',
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'is status 200': (r) => r.status === 200,
  });

  sleep(1);
}
