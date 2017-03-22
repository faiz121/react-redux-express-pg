import React from 'react';
import ReactDOM from 'react-dom';
import Home from './Home'

const App = React.createClass({
  render () {
    return (
        <div>
        <h1>Hello, world 2</h1>
        <Home />
        </div>
    )
  }
});

ReactDOM.render( <App />,
  document.getElementById('root')
);