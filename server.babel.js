import express from 'express';
import pool from './db';

pool.connect(function (err, client, done) {
    if (err) {
        console.log("Cannot connect to the DB" + err);
    }
    client.query('', function (err, result) {
        done();
        if (err) {
            console.log(err);
        }
        console.log('rows', result.rows);
    })
})
const app = express();
const morgan = require('morgan');
const bodyParser = require('body-parser');
const port = process.env.PORT || 3002;

app.use(morgan('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({'extended': 'true'}));
app.use('/', express.static('public'));

app.use(function(err, req, res, next){
  console.log('Something failed');
  res.status(500).send({"Error" : err.message})
});

app.listen(port, () => {
  console.log(`listening on port ${port}`);
});
