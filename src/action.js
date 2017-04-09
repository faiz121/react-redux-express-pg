import axios from 'axios';
export function setSearchTerm(searchTerm) {
  console.log('1. action creater called');
  return { type: 'SET_SEARCH_TERM', searchTerm };
}

export function addTodo(todo) {
  return { type: 'ADD_TODO', todo };
}

export function removeTodo(id) {
  return { type: 'REMOVE_TODO', id };
}

export function getTodosFromDB() {
  return function (dispatch, getState) {
    console.log('in getTodosFromDB');
    axios.get('/api/todos/')
        .then((res) => {
          dispatch(addTodo(res.data));
        })
        .catch((error) => console.error('axios error', error));
  }
}

export function postTodoToDB(task) {
  return function (dispatch, getState) {
    console.log('in getTodosFromDB');
    axios.post('/api/todos/', {"task" : task})   //[{"_id":"58d346b2f70e8d47a740112e","task":"get milk","__v":0}]
        .then((res) => {
          console.log(`addTodo res ${JSON.stringify(res.data)}`);
          dispatch(addTodo(res.data));
        });
  }
}
export function removeTodoToDB(id) {
  return function (dispatch, getState) {
    axios.delete('/api/todos/' + id)
        .then((res) => {
          dispatch(removeTodo(id));
        });
  }
}
