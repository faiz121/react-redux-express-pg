const mongoose = require('mongoose');

mongoose.connect('mongodb://'+process.env.dbuser+':'+process.env.dbpass+'@ds161225.mlab.com:61225/faiztestdb123');
// mongoose.connect('mongodb://localhost/faiztestdb');
mongoose.connection.once('open', function () {
  console.log('Connected to db');
}).on('error', function () {
  console.log(`-----------------------------------------------------`);
  console.log('Error connecting to db');
  console.log(`-----------------------------------------------------`);
  console.log(`Make sure you are connecting to your deployed mongo instance or 
   if it is local then do not forget to run 'mongod' to start your local instance
   --- make db config changes in /db.js`);
});
