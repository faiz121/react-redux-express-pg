import pg from 'pg';
const config = {
  user: 'postgres',
  database: 'dvdrental',
  password: 'test1234',
  port: 5433
};

const pool = new pg.Pool(config);
export default pool;