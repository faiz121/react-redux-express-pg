import React from 'react';
import ReactDOM from 'react-dom';
import '../public/style.css';
import axios from 'axios';
import store from './store';
import {Provider} from 'react-redux';
import DrawCanvas from './DrawCanvas';
import NetStatistics from './NetStatistics';

const App = React.createClass({
  componentDidMount () {
  },
  render () {
    return (
        <Provider store={store}>
          <div className="container">
            <DrawCanvas />
            <NetStatistics name={"cat"} nodeValues={["cat"]} prediction={1} />
          </div>
        </Provider>
    )
  }
});

ReactDOM.render(<App />,
    document.getElementById('app')
);
