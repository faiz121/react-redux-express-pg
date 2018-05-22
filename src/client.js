import React from 'react';
import ReactDOM from 'react-dom';
import Home from './Home'
import '../public/style.css'
import axios from 'axios'
import store from './store'
import {Provider} from 'react-redux'
import Todo from './Todo'
import { getTodosFromDB } from './action'
import DrawCanvas from './DrawCanvas'

const App = React.createClass({
  componentDidMount () {
    store.dispatch(getTodosFromDB());
  },
  render () {
    return (
        <Provider store={store}>
          <div className="container">
            <h1 className="text-center">To Do List</h1>
            <Home/>
            <Todo />
            <DrawCanvas />
          </div>
        </Provider>
    )
  }
});

ReactDOM.render(<App />,
    document.getElementById('app')
);
