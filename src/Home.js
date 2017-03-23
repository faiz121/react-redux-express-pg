import React from 'react'

const Home = React.createClass({
  render () {
    let input;
    const { addTodo } = this.props;
    return (
        <div className="text-center">
          <input ref={(node) => input=node } type="text"  name="todo" placeholder='add a todo ...' />
          <button  className="btn btn-primary btn-sm" onClick={() => {
            addTodo(input.value);
            input.value = '';
          }} type="submit">add</button>
        </div>
    )
  }
});

export default Home;