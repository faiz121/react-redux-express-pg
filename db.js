import pg from 'pg';
const config = {
  user: 'perry',
  database: 'digits',
  password: '',
  port: 5432
};

const pool = new pg.Pool(config);
export default pool;
