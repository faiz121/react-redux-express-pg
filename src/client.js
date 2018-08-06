import React from 'react';
import ReactDOM from 'react-dom';
import '../public/style.css'
import axios from 'axios'
import store from './store'
import {Provider} from 'react-redux'
import DrawCanvas from './DrawCanvas'

const App = React.createClass({
  componentDidMount () {
  },
  render () {
    return (
        <Provider store={store}>
          <div className="container">
            <DrawCanvas />
          </div>
        </Provider>
    )
  }
});

ReactDOM.render(<App />,
    document.getElementById('app')
);
