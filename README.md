# React-TodoApp-Express-Mongo


### Tech Stack

1. React
2. Node.js & Express
3. MongoDB & Mongoose
4. Webpack (bundling)
5. BootStrap for Styling

![todo App](https://image.ibb.co/kCJZDv/Screen_Shot_2017_03_23_at_11_25_27_AM.png)

### Instructions to setup

##### Prerequisites
1. Node version > 4
2. mongodb installed

##### Steps
1. `git clone https://github.com/faiz121/React-TodoApp-Express-Mongo.git`
2. `npm install`
3. `mongod` (to start up mongodb instance)
4. `npm run dev`
5. Goto `http://localhost:3000`

##### Understanding the app

###### First and foremost Webpack Config

```javascript
{
  entry: './src/client.js',
  devtool: 'eval',
  output: {
    path: path.join(__dirname, '/public'),
    filename: 'bundle.js'
}
```
In the first part of `webpack.config.js` we specify what the entry point of our app is `client.js` ,  the parent component of your app which
has many child components. 
`devtool` is the type of debugging tool for a React app. There are a few but it may not be important at this point.
`output` is where the React app is transpiled and bundled and placed.

```javascript
loaders: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        loader: 'babel-loader'
      }
```
Here we say every `.js` file should be transpiled using babel-loader and the rest of the config says helps us load `.json` and `.css` files 
in any `.js`. 


