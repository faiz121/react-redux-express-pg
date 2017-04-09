import React from 'react'
import { connect } from 'react-redux'
import { setSearchTerm, postTodoToDB } from './action'

const Home = React.createClass({

  addTodo (task) {
    console.log(`addTodo ${task}`);
    this.props.dispatchAddTodo(task);
  },
  render () {
    let input;
    return (
        <div className="text-center">
          <input ref={(node) => input = node } type="text" name="todo" placeholder='add a todo ...'/>
          <button className="btn btn-primary btn-sm" onClick={() => {
            this.addTodo(input.value);
            input.value = '';
          }} type="submit">add
          </button>
          <div>
            <input type="text"
                   onChange={(e) => {
                     this.props.dispatchSetSearchTerm(e.target.value)
                   }}
                   value={this.props.searchTerm}
                    placeholder='search ...'/>
          </div>
        </div>
    )
  }
});

const mapStateToProps = (state) => {
  return {
    searchTerm: state.searchTerm
  }
};

const mapDispatchToProps = (dispatch) => {
  console.log(`setSearchTerm ${setSearchTerm}`);
  return {
    dispatchSetSearchTerm (searchTerm) {
      dispatch(setSearchTerm(searchTerm)) // dispatch({ type: 'SET_SEARCH_TERM', searchTerm = searchTerm })
    },
    dispatchAddTodo(todo) {
      dispatch(postTodoToDB(todo))
    }
  }
};

export default connect(mapStateToProps, mapDispatchToProps)(Home)
